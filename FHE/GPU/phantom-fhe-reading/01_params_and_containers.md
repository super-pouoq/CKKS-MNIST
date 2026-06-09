# 第 1 章　参数与密文容器

> 涉及文件：`host/encryptionparams.h`、`host/modulus.{h,cu}`、`host/hestdparms.h`、
> `plaintext.h`、`ciphertext.h`、`common.h`
>
> 本章是“数据结构”章：先把 **参数（怎么描述一套加密方案）** 和 **三大容器
> （Modulus / Plaintext / Ciphertext）** 讲清楚，后面所有算法都在操作它们。

---

## 1.1 这些文件是干什么的

| 文件 | 职责 |
|------|------|
| `host/modulus.{h,cu}` | 单个素数模数 `Modulus`，预存 Barrett 约简常数；`CoeffModulus` 工厂（造模数链） |
| `host/hestdparms.h` | HE 标准安全表：给定 N 返回模数总位宽上限（tc128/192/256） |
| `host/encryptionparams.h` | `EncryptionParameters`：方案/N/模数链/明文模数/特殊模数 等“配方” |
| `plaintext.h` | `PhantomPlaintext`：编码后的明文多项式（GPU 上的 `uint64_t*`） |
| `ciphertext.h` | `PhantomCiphertext`：密文（2~3 个多项式）+ 元数据（scale、level、是否 NTT 形式…） |
| `common.h` | 全局常量（block 维度、2^64 等），GPU kernel 调参 |

---

## 1.2 `Modulus`：一个带“加速器”的素数

`Modulus` 表示模数链里的**一个**小素数 q_i（≤ 61 bit）。它不仅存值，还预存了
**Barrett 约简**所需常数，这样后续每次取模都能避免昂贵的除法。

核心成员：
- `value_`：模数值。
- `bit_count_`：有效位数。
- `const_ratio_[3]`：Barrett 比率 = ⌊2^128 / value⌋（前两个 64-bit 字）+ 余数（第三字）。
- `is_prime_`：是否素数。

### `Modulus::set_value(uint64_t value)`（`modulus.cu`）
**做什么**：设置模数并算好加速常数。
**怎么做**：
1. 合法性检查——`value` 不能是 1，也不能超过 61 bit（`MOD_BIT_COUNT_MAX`），否则抛异常。
2. `bit_count_ = get_significant_bit_count(value_)`。
3. 关键一步：用 192-bit 长除法算 Barrett 比率。把 `numerator = 2^128`（写成 `{0,0,1}`
   这个 192-bit 数）除以 `value_`，商存进 `const_ratio_[0..1]`，余数存 `const_ratio_[2]`。
   这就是 `divide_uint192_inplace(numerator, value_, quotient)`（见附录 B）。

> **为什么要 2^128/value**：Barrett 约简用 `x mod q ≈ x - ⌊x·μ/2^128⌋·q`（μ 即此比率），
> 把“除以 q”换成“乘 μ + 移位”，GPU 上极快。这是整个库模运算的地基。

其余都是只读访问器：`value()`、`bit_count()`、`const_ratio()`、`is_prime()`、比较运算符等。

---

## 1.3 `CoeffModulus`：模数链工厂

`CoeffModulus`（`modulus.h`，实现散落在 `modulus.cu`/`numth.cu`）是个静态工具类。

### `CoeffModulus::MaxBitCount(N, sec_level)`
**做什么**：返回在多项式次数 N、给定安全级别下，**模数链总位宽的上限**。
**怎么做**：直接查 `hestdparms.h` 里的标准表（`he_std_parms_128_tc(N)` 等）。
例如 N=2^14 → 438 bit，N=2^15 → 881 bit。超过会降低安全性，所以参数设置时要守住它。

### `CoeffModulus::Create(N, bit_sizes)`
**做什么**：按你要的每个素数的位宽（如 `{60,40,40,40,60}`），生成一串
**NTT-friendly 素数**（满足 q ≡ 1 mod 2N，且互不相同）。
**怎么做**：调用数论工具 `get_primes(...)`（`numth.cu`），对每个目标位宽，从 `2^bits` 附近
往下找满足 `≡ 1 (mod 2N)` 的素数（Miller-Rabin 判素）。返回 `vector<Modulus>`。

> 第一个/最后一个常取 60-bit（首项放精度余量、末项作 special modulus），中间的 40-bit
> 个数 ≈ 可用乘法深度——这正是 ckks-demo 里 `{60,40,40,40,60}` 的含义。

---

## 1.4 `EncryptionParameters`：一套方案的“配方”

这是用户唯一要手填的对象。成员：
- `scheme_`：`bgv` / `bfv` / `ckks`。
- `poly_modulus_degree_`（N）：必须是 2 的幂。
- `coeff_modulus_`：当前模数链（计算中会随 rescale 变短）。
- `key_modulus_`：**完整**模数链（含 special modulus），key-switch 用；第一次 `set_coeff_modulus`
  时把它拷成 `key_modulus_`，之后 coeff 变短而 key 保持。
- `special_modulus_size_`：hybrid key-switching 的特殊模数 P 的个数（默认 1）。
- `plain_modulus_`：明文模数（仅 BFV/BGV 用，CKKS 不用）。
- `mul_tech_`：BFV 的乘法技术（behz/hps/...）。

关键 setter 的“怎么做”：
- `set_poly_modulus_degree(N)`：仅在方案非 none 时允许非零 N；只存值，合法性（2 的幂）留给 context。
- `set_coeff_modulus(v)`：检查个数在 `[COEFF_MOD_COUNT_MIN, MAX]`；**首次**设置时把 `v` 同时写入
  `key_modulus_`（完整链）；之后只更新 `coeff_modulus_`。
- `set_plain_modulus(t)`：非 BFV/BGV 且 t≠0 时报错（CKKS 不该设明文模数）。
- `set_special_modulus_size(k)`、`set_mul_tech(...)`：同理做方案合法性校验。

> 这个类**只描述配方，不做预计算**。真正把素数链、NTT 表、各种降基矩阵算出来的是第 2 章的 `Context`。

---

## 1.5 `PhantomPlaintext`：编码后的明文多项式

CKKS 的明文不是“一个数”，而是 **一个多项式**（已 encode + 乘 scale 取整）。成员：
- `chain_index_`：所在层级（决定用哪一段模数链）。
- `poly_modulus_degree_`、`coeff_modulus_size_`：多项式形状（N × 模数个数）。
- `scale_`：缩放因子（CKKS 关键，decode 时要除回去）。
- `data_`：`cuda_auto_ptr<uint64_t>`，GPU 上 `coeff_modulus_size × N` 个系数（RNS 形式）。

方法都很“朴素”：
- `resize(coeff_modulus_size, N, stream)`：在 GPU 上分配 `coeff_modulus_size*N` 个 uint64 并记录形状。
- `coeff_count()`：返回 `N × coeff_modulus_size`（总系数个数）。
- `save/load`：把元数据写流，再把 GPU 数据拷到 pinned host 内存（`cudaMallocHost`）后写出/读入。
- 访问器：`scale()`、`chain_index()`、`data()`。

---

## 1.6 `PhantomCiphertext`：密文

密文 = `size_` 个多项式（新鲜/加法后为 2：c0,c1；密文乘后暂为 3：c0,c1,c2）。成员（`ciphertext.h`）：
- `chain_index_`、`size_`、`poly_modulus_degree_`、`coeff_modulus_size_`：形状与层级。
- `scale_`：CKKS 缩放因子。
- `correction_factor_`：**BGV** 解密用的校正因子（CKKS=1）。
- `noiseScaleDeg_`：缩放因子的“次数”，乘法/rescale 时跟踪。
- `is_ntt_form_`：当前是否在 NTT 域（CKKS 密文默认 true）。
- `is_asymmetric_`：公钥加密(true) 还是对称加密(false)。
- `data_`：GPU 上 `size × coeff_modulus_size × N` 个 uint64，布局 `[poly][modulus][coeff]`。
- `seed_`：仅对称加密时存（用种子复现 c1，省一半存储）。

方法的“怎么做”：

### `resize(context, chain_index, size, stream)`
**做什么**：把密文重新分配成 `size` 个多项式、对齐到 `chain_index` 层。
**怎么做**：从 `context.get_context_data(chain_index)` 取出该层的模数个数与 N，算新总长度；
若与旧长度不同则重新 `make_cuda_auto_ptr`，并 `cudaMemcpyAsync` 把旧数据（取 `min(old,new)`）拷过来
——所以**变大时保留原数据**（乘法把 2 项扩到 3 项时用得上）。最后更新形状字段。
另一重载 `resize(size, coeff_modulus_size, N, stream)` 直接给形状、不查 context。

### setter / getter
全是平凡转发：`set_scale/set_chain_index/set_ntt_form/set_correction_factor/SetNoiseScaleDeg`，
以及对应 getter `scale()/chain_index()/size()/is_ntt_form()/correction_factor()/data()`。
这些字段是**同态运算的“账本”**：第 5 章 `add_aligned`、rescale 都靠读写它们维持正确性。

### `save_symmetric / load_symmetric`（见 `ciphertext.h` 末尾）
对称密文只存 c0 + 种子；`load_symmetric` 时用种子 `sample_uniform_poly_wrap` 重新生成 c1，
必要时再 `nwt_2d_radix8_backward_inplace` 变回非 NTT 形式。非对称密文则照常存两条多项式。

---

## 1.7 `common.h`：GPU 调参常量

- `blockDimGlb(128)`：逐系数 kernel 的线程块大小（必须整除 N）。
- `gridDimNTT/blockDimNTT/per_thread_sample_size=8/per_block_pad=4`：NTT kernel 的网格配置
  （radix-8，每线程处理 8 个样本，见附录 A）。
- `two_pow_64`、`n_cuda_streams=10` 等：浮点常量与多流并发数。

---

## 1.8 函数速查表

| 符号 | 文件 | 作用 |
|------|------|------|
| `Modulus::set_value` | modulus.cu | 设模数 + 算 Barrett 比率 |
| `CoeffModulus::MaxBitCount` | modulus.h | 安全位宽上限（查表） |
| `CoeffModulus::Create` | modulus.cu | 生成 NTT-friendly 素数链 |
| `EncryptionParameters::set_*` | encryptionparams.h | 填方案配方（不预计算） |
| `PhantomPlaintext::resize/save/load` | plaintext.h | 明文多项式分配/序列化 |
| `PhantomCiphertext::resize` | ciphertext.h | 密文按层级/项数分配，变大保留旧数据 |
| `PhantomCiphertext::set_*/getters` | ciphertext.h | scale/level/ntt 等“账本”读写 |

下一章进入真正的“引擎室”：RNS 工具与 `Context` 预计算。
