# 第 3 章　CKKS 编码器（encode / decode）

> 涉及文件：`ckks.{h,cu}`、`fft.{h,cu}`；附 `batchencoder.{h,cu}`（BFV/BGV 的打包编码）。
>
> CKKS 的“魔法”在编码：它把一串**复数/实数**塞进一个**整系数多项式**，使得
> “多项式乘法 ≈ 向量逐元素乘”，从而支持 SIMD 风格的密文并行计算。本章讲清这条
> `复数向量 → (逆FFT) → 整系数 → (RNS+NTT) → 明文多项式` 的来回。

---

## 3.1 数学直觉（30 秒版）

- 槽数 `slots = N/2`。CKKS 用 X^N+1 的复根做一个“变种 DFT”（`special_fft`）。
- **encode**：向量 → 逆变换得到长度 N 的复系数 → 乘 `scale` 取整 → 得整系数多项式。
- **decode**：整系数多项式 → 除以 `scale` → 正变换 → 取前 `slots` 个复数。
- 因为根的 4 重对称性，只需存 `n/8` 个根（`ComplexRoots`），且变换在 `slots` 上做（不是 N）。

---

## 3.2 `DCKKSEncoderInfo` 与 `ComplexRoots`（`fft.h`）

- **`ComplexRoots(degree)`**：预计算 0~(n/8−1) 次 n 次本原根，靠 4 重对称性 `get_root(i)` 推出任意根。
- **`DCKKSEncoderInfo`**：GPU 上的工作缓冲：
  - `in_`：长度 `slots` 的复数输入/输出缓冲；
  - `twiddle_`：长度 `m=2N` 的 FFT 旋转因子表；
  - `mul_group_`：长度 `slots/2` 的“旋转群”下标（决定 slot 的排列顺序）。

---

## 3.3 构造 `PhantomCKKSEncoder`（`ckks.cu`）

`PhantomCKKSEncoder(context)` 做了：
1. 取首层参数，校验是 CKKS；令 `slots_ = N/2`，`m = 2N`。
2. 建 `gpu_ckks_msg_vec_`（`DCKKSEncoderInfo`）。
3. **算旋转群**：`pos` 从 1 开始反复 `pos = pos*5 mod m`，得到 `rotation_group_[i]`。
   生成元 5 是 Z*_{2N} 里的标准选择，决定了 slot ↔ 系数 的对应关系（也决定了第 6 章 rotate 的步长语义）。
4. **算根表**：`m≥8` 时用 `ComplexRoots` 填满 `root_powers_[0..m-1]`；`m==4` 直接给 {1,i,-1,-i}。
5. 把根表和旋转群 `cudaMemcpyAsync` 到 GPU。

辅助 kernel **`bit_reverse_kernel`**：把数组按 `log_n` 位做比特反转重排（FFT 前后都要用）。

---

## 3.4 `encode_internal`：向量 → 明文多项式

签名：`encode_internal(context, values(复数), chain_index, scale, dest, stream)`。
（用户调 `encode(values, scale, dest)` 时，实数会先被包成 `{x,0}` 的复数。）

逐步“怎么做”：
1. **校验**：非空、`values.size() ≤ slots`、`scale` 为正且 `log2(scale)+1 < 总模数位宽`（否则溢出）。
2. 把 `values` 拷到 GPU `temp`，把 `in_` 清零。
3. `bit_reverse_kernel`：把输入按 `log_slot_count` 比特反转放进 `in_`（FFT 要求）。
4. 令 `fix = scale / slots`，调 **`special_fft_backward(*gpu_ckks_msg_vec_, log_slot, fix, stream)`**：
   做逆 special-FFT，同时把结果乘上 `fix`——这一步把复数向量变成**实/复系数**并施加缩放。
5. 把结果拷回 host，扫描最大系数绝对值 `max_coeff`，算 `max_coeff_bit_count`；
   若 ≥ 总模数位宽则抛“encoded values are too large”（精度/溢出守门）。
6. **`rns_tool.base_Ql().decompose_array(dest.data(), in_, coeff_count, max_bit, stream)`**：
   把这组（取整后的）大整系数拆成 RNS 形式写入明文（第 2 章的 `DRNSBase::decompose_array`）。
7. **`nwt_2d_radix8_forward_inplace(dest, gpu_rns_tables, coeff_modulus_size, 0, stream)`**：
   对每个 RNS 分量做正向 NTT——明文最终以 **NTT 域** 存放（这样后面与密文逐点相乘）。
8. 记录 `dest.chain_index_ = chain_index`、`dest.scale_ = scale`。

> 注意第 4 步是**逆**变换（backward）用于 encode，第 6 章解密后 decode 用**正**变换——
> 这与普通 DFT 的“编码用逆、解码用正”约定一致。

---

## 3.5 `decode_internal`：明文多项式 → 向量

签名：`decode_internal(context, plain, dest(复数), stream)`。

1. 校验 `scale`；准备 `upper_half_threshold`（区分系数正负的分界，拷到 GPU）；`in_` 清零。
2. `inv_scale = 1/plain.scale()`；把明文数据拷一份 `plain_copy`（不破坏原明文）。
3. **`nwt_2d_radix8_backward_inplace(plain_copy, gpu_rns_tables, ...)`**：逆 NTT，回到系数域。
4. **`rns_tool.base_Ql().compose_array(in_, plain_copy, upper_half_threshold, inv_scale, coeff_count, stream)`**：
   CRT 重建大整系数 → 用阈值中心化到 (−Q/2, Q/2) → 乘 `inv_scale` 得到复系数（第 2 章 `compose_array`）。
5. **`special_fft_forward(*gpu_ckks_msg_vec_, log_slot, stream)`**：正向 special-FFT，复系数 → 槽值。
6. `bit_reverse_kernel` 把结果反转回正常顺序，取前 `slots` 个拷回 host。
7. 用户层 `decode<double>` 再取每个复数的实部 `.x`。

---

## 3.6 `special_fft_forward / backward`（`fft.cu`）

这两个函数是 GPU 版“变种 FFT”。实现用了**两段式**：
- `inplace_special_ffft_base_kernel` / `inplace_special_ifft_base_kernel`：处理最初几层（数据在
  一个 block 内、用 shared memory 完成多级蝶形）。
- `..._iter_kernel`：处理后续大跨度层，每层一次 kernel，按 `mul_group_`/`twiddle_` 做复数蝶形。
- `backward` 版额外接收 `scalar`（即 `fix=scale/slots`），在最后把每个元素乘上它，省一次遍历。

> 你可以把 `special_fft` 当成“CKKS 专用 DFT”：和附录 A 的 NTT 是**两套独立的变换**——
> special-FFT 作用在**复数槽 ↔ 复系数**（编码层面），NTT 作用在**整系数 ↔ 点值**（模 q_i 层面）。

---

## 3.7 附：`PhantomBatchEncoder`（BFV/BGV，`batchencoder.{h,cu}`）

CKKS 用复数 FFT，而 BFV/BGV 是**整数明文**，用的是基于明文模数 t 的 NTT 打包：
- 构造时 `populate_matrix_reps_index_map`：用 2N 阶生成元算出“矩阵表示”的 slot 排列下标。
- `encode(vector<uint64_t>)`：按下标把整数放进系数槽，再做**明文 NTT**（`gpu_plain_tables`）。
- `decode`：逆明文 NTT，再按下标取回整数。
没有 scale、没有复数，比 CKKS 简单；本书主线是 CKKS，这里只作对照。

---

## 3.8 函数速查表

| 符号 | 文件 | 作用 |
|------|------|------|
| `PhantomCKKSEncoder::ctor` | ckks.cu | 算旋转群 + 根表，建 GPU 缓冲 |
| `encode_internal` | ckks.cu | 向量→逆FFT→取整→RNS分解→正NTT |
| `decode_internal` | ckks.cu | 逆NTT→CRT合成→正FFT→取槽值 |
| `bit_reverse_kernel` | ckks.cu | 比特反转重排 |
| `special_fft_forward/backward` | fft.cu | CKKS 专用复数 FFT（GPU） |
| `ComplexRoots::get_root` | fft.cu | 4 重对称取本原根 |
| `PhantomBatchEncoder::encode/decode` | batchencoder.cu | BFV/BGV 整数打包（对照） |

下一章：明文如何被加密成密文，以及密钥怎么来。
