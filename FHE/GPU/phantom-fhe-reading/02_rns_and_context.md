# 第 2 章　RNS 与上下文预计算

> 涉及文件：`host/rns.{h,cu}`（CPU 端 `RNSBase`/`RNSNTT`）、`rns_base.{cuh,cu}`（`DRNSBase`）、
> `rns.{cuh,cu}`（`DRNSTool`，GPU 端 RNS 引擎）、`rns_bconv.{cuh,cu}`（基转换 kernel）、
> `context.{cuh,cu}`（`ContextData` / `PhantomContext`）。
>
> 这是整个库的“引擎室”。**RNS（剩余数系统）** 决定了数据如何在 GPU 上排布，
> **Context** 则在构造时把所有层级、所有降基/升基所需的常量一次性预计算好。

---

## 2.1 RNS 是什么、为什么用它

一个系数可能有几百比特，GPU 原生只擅长 64-bit。CRT（中国剩余定理）告诉我们：
若 q = q_0·q_1·…·q_{k-1}（互素），则一个 mod q 的大整数 x 可以**唯一**地用
`(x mod q_0, x mod q_1, …)` 这一组小余数表示，且加减乘都可以**逐分量并行**完成。

于是 Phantom 把每个多项式存成 `k` 段“小多项式”，每段系数都 < 2^61。GPU 一个线程管一个
`(modulus_idx, coeff_idx)`，天然并行。**代价**是某些操作（rescale、解密、key-switch）需要在
不同 RNS 基之间“换算”，这就是 **base conversion（基转换）**，也是本章的重头戏。

涉及的预计算量（都挂在 `DRNSBase` / `DRNSTool` 里）：
- `big_Q_`：完整模数乘积 Q。
- `qiHat_ = Q/q_i`，`qiHatInv_mod_qi_ = (Q/q_i)^{-1} mod q_i`：CRT 重建系数。
- `qiInv_`：1/q_i 的浮点近似，用于“浮点法”快速估 CRT 进位。
- 各种 `..._shoup_`：Shoup 预乘常量（让“模乘一个固定常数”免做 128-bit 约简）。

---

## 2.2 `RNSBase`（CPU 端，`host/rns.{h,cu}`）

`RNSBase` 是模数链的 CPU 表示，负责**算出**上面那些 CRT 常量，再由 GPU 端拷走。

关键方法：
- 构造 `RNSBase(vector<Modulus>)`：存模数，校验两两互素。
- `decompose(uint64_t* value)`：把一个“大整数”（占 k 个字）就地拆成 RNS 余数
  `value[i] = bigint mod q_i`。Context 里把 BFV 的 Δ、`upper_half_increment` 拆成 RNS 就靠它。
- `compose(...)`：逆操作，用 CRT 把 RNS 余数重建回大整数。
- 内部预计算 `Q`、`q_iHat`、`q_iHatInv mod q_i` 等。

`RNSNTT`：对模数链里每个素数各持一份 `NTT` 表（见附录 A），供 forward/inverse NTT 用。

---

## 2.3 `DRNSBase`（GPU 端，`rns_base.{cuh,cu}`）

`DRNSBase::init(const RNSBase& cpu, stream)` 把上面 CPU 算好的常量 `cudaMemcpy` 到显存：
`base_`（各 `DModulus`）、`big_Q_`、`big_qiHat_`、`qiHat_mod_qi_`(+shoup)、`qiHatInv_mod_qi_`(+shoup)、`qiInv_`。

它额外提供两个 CKKS 编码直接要用的 kernel 封装：
- **`decompose_array(dst, src(复数取整后的大整数), coeff_count, max_bit, stream)`**：
  把“编码得到的、最多 `max_coeff_bit_count` 位的整系数”拆成 RNS 形式写进 `dst`。
  这是 **encode 的最后一步**（第 3 章）。
- **`compose_array(dst(复数), src(RNS), upper_half_threshold, inv_scale, coeff_count, stream)`**：
  把 RNS 系数 CRT 重建成大整数 → 按 `upper_half_threshold` 判正负（中心化到 (-Q/2, Q/2)）
  → 乘 `inv_scale` 还原成复数。这是 **decode 的第一步**（第 3 章）。

---

## 2.4 `DRNSTool`（GPU 端 RNS 引擎，`rns.{cuh,cu}`）

这是最核心也最庞大的类（`rns.cu` 有 10 万字符）。一个 `DRNSTool` 绑定**某一层** Q_l，
持有该层做基转换/rescale/降基所需的全部常量，并暴露成一组 GPU 方法。

成员（节选，名字即含义）：
- key-switch 相关：`bigP_mod_q_`、`bigPInv_mod_q_`（特殊模数 P 的处理）、
  `partQlHatInv_mod_Ql_concat_`（modup 的分块 CRT 常量）。
- rescale 相关：`q_last_mod_q_`、`inv_q_last_mod_q_`（“除掉最后一个素数”的常量）。
- BFV 乘法相关：`gpu_Bsk_tables_`、`tModBsk_`、`inv_prod_q_mod_Bsk_` 等（BEHZ/HPS 基 Bsk）。

最重要的成员函数（“做什么/怎么做”）：

### `modup(dst, in, ntt_tables, ...)`
**做什么**：把一段系数从基 Q_l **升基**到 Q_l∪P（hybrid key-switching 的第一步）。
**怎么做**：对每个 RNS 分量做 `q_iHatInv` 缩放，再用 `QHatModp` 把它“投影”到目标素数上累加
（即 fast base conversion），最后对新增的模数做 NTT。实现分散在 `ntt/ntt_modup.cu`（第 6 章详述）。

### `moddown(...)` / `moddown_from_NTT(...)`
**做什么**：key-switch 末尾把结果从 Q_l∪P **降基**回 Q_l（除掉 P）。
**怎么做**：先把 P 部分基转换到 Q_l，做差再乘 `bigPInv_mod_q`（即除以 P）。见 `ntt/ntt_moddown.cu`。

### `divide_and_round_q_last_ntt(src, size, ntt_tables, ...)`
**做什么**：CKKS **rescale** 的核心——把密文除以最后一个素数 q_last 并四舍五入。
**怎么做**：把最后一个 RNS 分量 `INTT` 回系数域，广播减到其余分量上，再乘 `inv_q_last_mod_q`，
对每个 q_i 完成“整除并就近取整”。结果层级 -1。

### `behz_decrypt_scale_and_round / hps_decrypt_scale_and_round`
**做什么**：BFV 解密里把 mod-Q 的结果缩放回 mod-t 并取整（两种乘法技术各一套）。

### `fastbconv_m_tilde / sm_mrq / fast_floor / fastbconv_sk`
**做什么**：BFV **BEHZ 乘法**用到的一串基转换原语（引入 m̃、辅助基 B、求 floor、最终回 q）。

> 你不需要记住每一个常量；记住一句话即可：**任何“跨 RNS 基”的换算（rescale、key-switch 的
> up/down、BFV 乘法、解密取整）都由 `DRNSTool` 提供，常量在构造时一次算好。**

---

## 2.5 `rns_bconv.{cuh,cu}`：基转换 kernel

这里是上面那些方法真正落到 GPU 的地方。核心是 `base_convert_acc*` 系列设备函数（在 `rns.cuh`
末尾就能看到）：对每个输出素数 p，累加 `Σ_i src[i] · QHatModp[p][i]`（128-bit 累加避免溢出），
还提供 `unroll2/unroll4` 版本一次处理 2/4 个系数提升吞吐；`*_frac` 版本用 `double` 估算 CRT
进位（“浮点法 base conversion”，比纯整数快）。`add_to_ct_kernel` 等把转换结果加回密文。

---

## 2.6 `ContextData`：单层的预计算包（`context.cu`）

`ContextData` 对应模数链上的**一层**。构造函数 `ContextData(params, stream)` 做：
1. `multiply_many_uint64` 算出该层 `total_coeff_modulus_`(=∏q_i) 及其位宽（用于安全评估）。
2. 建 `RNSBase(coeff_modulus)` 与 `small_ntt_tables_`（每个素数一份 NTT 表）。
3. **BFV/BGV 分支**：算 Δ=⌊Q/t⌋（`coeff_div_plain_modulus_`）及余数 `upper_half_increment`，
   把它们 `decompose` 成 RNS；建明文 NTT 表；算 `plain_upper_half_threshold=(t+1)/2`。
4. **CKKS 分支**：要求 `plain_modulus==0`；设 `plain_upper_half_threshold = 2^63`
   （区分明文系数正负的分界），并对每个 q_i 算 `plain_upper_half_increment = 2^64 mod q_i`。
5. 建该层的 `gpu_rns_tool_`（`DRNSTool`）。

只读访问器把这些常量暴露给 encoder/evaluator/decryptor。

---

## 2.7 `PhantomContext`：把所有层串成链（`context.cu`）

构造 `PhantomContext(params)` 是“开机”动作：

1. 令 `size_QP = key_modulus.size()`，`size_P = special_modulus_size`，`size_Q = size_QP - size_P`。
2. **建链**：先 `emplace_back` 一个**含特殊模数 P 的**层（这是 key 层，`context_data_[0]`）；
   若 `size_P≠0` 标记 `using_keyswitching_=true`。
3. 然后 `pop_back` 掉所有 P，逐层 `emplace_back` 并每次再 `pop_back` 一个 q_i——
   于是 `context_data_[1]` 是“首个数据层”（满模数），往后每层少一个素数，直到最底层。
4. 给每层 `set_chain_index(idx)`。`first_parm_index_` 设为 1（若有 >1 层）。
5. 建 **GPU 端 NTT 表** `gpu_rns_tables_`：对每个素数把 root powers / inv root powers / inv_degree
   等从 CPU `small_ntt_tables` 拷进去（NTT 全靠这些旋转因子，见附录 A）。
6. BFV/BGV 时再建 `gpu_plain_tables_`、拷 `plain_modulus_` 等；BFV 还要拼出所有层的
   `coeff_div_plain_`（Δ）到一块连续显存。

关键访问器（后面各章频繁调用）：
- `get_context_data(index)` / `key_context_data()`(=[0]) / `first_context_data()`(=[1]) / `last_context_data()`。
- `get_context_data_rns_tool(index)`：取某层的 `DRNSTool`。
- `previous_parm_index/next_parm_index`：在链上上下移动（rescale=向 next 走）。
- `using_keyswitching()`：是否支持 relinearize/rotate（需 ≥2 个素数）。
- `gpu_rns_tables()` / `gpu_plain_tables()`：GPU NTT 表。

> 一句话总结：**`PhantomContext` 把“一条会随 rescale 变短的模数链”物化成一个数组
> `context_data_[]`，每个元素携带该层的全部 RNS/NTT 常量；`chain_index` 就是数组下标。**

---

## 2.8 函数速查表

| 符号 | 文件 | 作用 |
|------|------|------|
| `RNSBase::decompose/compose` | host/rns.cu | 大整数 ↔ RNS 余数（CRT） |
| `DRNSBase::init` | rns_base.cu | 把 CRT 常量拷到 GPU |
| `DRNSBase::decompose_array/compose_array` | rns_base.cu | encode/decode 的 RNS 步骤 |
| `DRNSTool::modup/moddown` | rns.cu | key-switch 升基/降基 |
| `DRNSTool::divide_and_round_q_last_ntt` | rns.cu | CKKS rescale 核心 |
| `DRNSTool::*decrypt_scale_and_round` | rns.cu | BFV 解密缩放取整 |
| `base_convert_acc*` | rns.cuh/rns_bconv.cu | 基转换累加 kernel |
| `ContextData::ContextData` | context.cu | 单层常量预计算 |
| `PhantomContext::PhantomContext` | context.cu | 建模数链 + GPU NTT 表 |

下一章：明文怎么从“一串实数”变成多项式——CKKS 编码器。
