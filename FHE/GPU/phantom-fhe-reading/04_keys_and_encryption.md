# 第 4 章　密钥生成与加密

> 涉及文件：`secretkey.{h,cu}`（私钥/公钥/relin/galois、加密、解密）、`prng.{cuh,cu}`（随机数）、
> `scalingvariant.{cuh,cu}`（BFV 的 Δ·m 缩放加法）。
>
> 本章把“钥匙串”和“加密”讲透。CKKS/BFV/BGV 的密钥结构其实**高度统一**：
> 公钥、relin key、galois key 本质都是“**对某个密钥的 RLWE 加密**”。

---

## 4.1 钥匙串总览（`secretkey.h`）

| 类 | 是什么 | 怎么存 |
|----|--------|--------|
| `PhantomSecretKey` | 私钥 s（及其幂 s,s^2,…） | `secret_key_array_`：连续存 s 的各次幂（NTT 域），`sk_max_power_` 记录已算到几次幂 |
| `PhantomPublicKey` | 公钥 (pk0,pk1) | 内部就是“0 的对称加密”，含 `prng_seed_a_`（c1 的种子） |
| `PhantomRelinKey` | 重线性化密钥 | 一组 `PhantomPublicKey`（key-switch key 的分块），`public_keys_ptr_` 是它们在 GPU 的指针数组 |
| `PhantomGaloisKey` | 旋转密钥 | 每个 galois 元素对应一个 `PhantomRelinKey`（即一把 key-switch key） |

> 记住这条主线：**relin key / galois key 都是 key-switch key**，区别只是“把哪个密钥切换回 s”。
> 第 6 章会看到它们如何被 `keyswitch_inplace` 统一消费。

---

## 4.2 私钥生成 `gen_secretkey`（`secretkey.cu`）

**做什么**：采样三元私钥 s ∈ {−1,0,1}^N，并存成 NTT 域。
**怎么做**：
1. 取 key 层（`context_data[0]`，满模数）的参数。
2. `random_bytes` 生成 PRNG 种子。
3. `sample_ternary_poly` kernel：在每个 q_i 下采样三元多项式到 `secret_key_array_`。
4. `nwt_2d_radix8_forward_inplace`：把 s 变到 NTT 域（之后所有运算都在 NTT 域用 s）。
5. `sk_max_power_ = 1`（目前只有 s^1）。

### `compute_secret_key_array(context, max_power)`
**做什么**：把私钥幂扩展到 s^{max_power}（relin 要 s^2、解密 size-3 密文要 s^2…）。
**怎么做**：分配更大数组、拷入已有幂，然后用 `multiply_rns_poly` 逐次 `s^{k}=s^{k-1}·s`（NTT 域逐点乘）。

---

## 4.3 “加密 0”——一切密钥/加密的原子操作

RLWE 的核心是“0 的加密”：给定随机 a 与小误差 e，`(−(a·s+e), a)` 解密回 0。
公钥、relin key、新鲜密文都建立在它之上。

### `encrypt_zero_symmetric`（私钥版）
**做什么**：用私钥造 `(c0,c1)=(−(a·s+e), a)`（BGV 是 `−(a·s+t·e)`）。
**怎么做**（NTT 形式分支）：
1. `cipher.resize(...,2,...)`；标记 NTT 形式、scale=1。
2. `sample_error_poly` 采样误差 e；`sample_uniform_poly` 用种子 `prng_seed_a` 生成 a 写进 c1。
3. BGV 时把 e 乘明文模数 t（`multiply_scalar_rns_poly`）。
4. e 做 NTT；`multiply_and_add_negate_rns_poly`：`c0 = −(c1·s + e)`。
非 NTT 分支则先 `c0=c1·s`，逆 NTT 回系数域再 `c0=−(c0+e)`，c1 也逆 NTT。

> 对称加密只需存 c0 + 种子（c1 可由种子复现），这就是第 1 章 `save_symmetric` 的依据。

### `encrypt_zero_asymmetric_internal_internal`（公钥版）
**做什么**：用公钥造 0 的加密 `c[j] = pk[j]·u + e[j]`（BGV 加 t）。
**怎么做**：采样三元 `u`（`sample_ternary_poly`）并 NTT；对每个多项式 j 采样误差 e[j]、NTT，
再 `multiply_and_add_rns_poly` 算 `u·pk[j]+e[j]`。非 NTT 分支对称处理。

---

## 4.4 公钥 / relin / galois 生成

### `gen_publickey`
就是“私钥加密一个 0”：随机种子 a → `encrypt_zero_symmetric(..., chain_index=0, is_ntt=true)`，
结果放进 `pk_`。所以**公钥 = 0 的对称密文**。

### `generate_one_kswitch_key`（私有，key-switch key 生成器）
**做什么**：给定“旧密钥 new_key”（如 s^2 或旋转后的 s），生成把它切换回 s 的 key-switch key。
**怎么做**：按 hybrid key-switching 的分块（dnum 块），每块造一个“`new_key·P_block` 的 RLWE 加密”，
存成一组 `PhantomPublicKey`。细节与 P（特殊模数）相关，第 6 章 key-switch 时配合阅读。

### `gen_relinkey`
**做什么**：生成把 s^2 切换回 s 的 key（密文乘法后用）。
**怎么做**：`compute_secret_key_array(...,2)` 确保有 s^2，取 `sk_square = secret_key_array + 偏移`，
调 `generate_one_kswitch_key(context, sk_square, relin_key)`。

### `create_galois_keys`
**做什么**：为每个 galois 元素生成一把旋转 key。
**怎么做**：对每个 `galois_elt`（必须奇数且 < 2N），用 `key_galois_tool->apply_galois_ntt` 把私钥 s
做“伽罗瓦置换”得到 `rotated_secret_key`，再 `generate_one_kswitch_key` 把它切换回 s。每把存进
`galois_keys.relin_keys_[i]`。（galois 元素↔旋转步数的映射见第 6 章。）

---

## 4.5 真正的加密 `encrypt_symmetric` / `encrypt_asymmetric`

加密 = “加密 0” + “把明文加进 c0”。

### CKKS（`encrypt_symmetric` 的 ckks 分支）
1. 用 `plain.chain_index()` 找到对应层，`encrypt_zero_symmetric(..., is_ntt=true)` 造 0 密文。
2. `add_rns_poly`：`c0 = c0 + plaintext`（明文已是 NTT 域，逐点加）。
3. `cipher.scale_ = plain.scale()`。
非对称版 `PhantomPublicKey::encrypt_asymmetric`：先 `encrypt_zero_asymmetric_internal` 造 0，再加明文。

### BFV（`encrypt_symmetric` 的 bfv 分支）
造 0 密文后调 `multiply_add_plain_with_scaling_variant`（`scalingvariant.cu`）：把明文乘 Δ=⌊Q/t⌋
再加到 c0——这是 BFV “整数明文嵌进高位”的关键缩放。

### BGV 分支
明文先复制到各 RNS 分量、做 NTT，再 `add_rns_poly` 加到 c0（BGV 明文在低位，误差是 t·e）。

---

## 4.6 CKKS 解密 `ckks_decrypt`

**做什么**：算 `m ≈ c0 + c1·s (+ c2·s^2 …)`，得到（仍是 NTT 域的）明文多项式。
**怎么做**：
1. 要求密文是 NTT 形式；`needed_sk_power = size−1`，必要时 `compute_secret_key_array` 补足 s 的幂。
2. `destination = c0`（拷贝）。
3. 循环 `i=1..size−1`：`multiply_and_add_rns_poly` 做 `dest += c_i · s^i`（NTT 域逐点）。
4. 设 `dest.chain_index = 密文层`、`dest.scale = 密文scale`。
之后用户调第 3 章的 `decode` 把这个明文多项式变回复数向量。

> CKKS 解密**不取整、不缩放**——噪声直接体现为 decode 后的微小误差，这正是 CKKS“近似计算”的本质。
> 而 `bfv_decrypt` / `bgv_decrypt` 多了一步“缩放回 mod t 并取整”（用第 2 章 `DRNSTool` 的
> `*decrypt_scale_and_round` / `divide_and_round_q_last`），因为它们是**精确整数**方案。
> `decrypt(...)` 是统一入口，按 `scheme` 分派到三者之一。

---

## 4.7 随机数与缩放变体

- **`prng.{cuh,cu}`**：`random_bytes` 生成种子；`sample_ternary_poly`（私钥/u）、`sample_error_poly`
  （CBD 离散高斯近似误差）、`sample_uniform_poly`（均匀 a）等 kernel。底层用 blake2/CUDA PRNG
  （见附录 C），保证可由种子复现（对称密文省存储）。
- **`scalingvariant.{cuh,cu}`**：`multiply_add_plain_with_scaling_variant`——BFV 专用，
  把明文系数乘 Δ 加到密文 c0，处理“负系数要补 r_t(q)”的细节（第 2 章 `upper_half_increment`）。

---

## 4.8 函数速查表

| 符号 | 文件 | 作用 |
|------|------|------|
| `gen_secretkey` | secretkey.cu | 采样三元私钥 + NTT |
| `compute_secret_key_array` | secretkey.cu | 扩展 s 的幂 |
| `encrypt_zero_symmetric/asymmetric*` | secretkey.cu | RLWE “0 的加密”原子操作 |
| `gen_publickey` | secretkey.cu | 公钥 = 0 的对称密文 |
| `generate_one_kswitch_key` | secretkey.cu | 造 key-switch key（relin/galois 共用） |
| `gen_relinkey/create_galois_keys` | secretkey.cu | 重线性化/旋转密钥 |
| `encrypt_symmetric/encrypt_asymmetric` | secretkey.cu | 加密 = 加密0 + 加明文 |
| `ckks_decrypt/bfv_decrypt/bgv_decrypt/decrypt` | secretkey.cu | 解密（CKKS 不取整） |
| `sample_*_poly` | prng.cu | 三元/误差/均匀采样 |
| `multiply_add_plain_with_scaling_variant` | scalingvariant.cu | BFV Δ·m 缩放加法 |

下一章：密文之间怎么加、怎么乘、怎么 relinearize 与 rescale。
