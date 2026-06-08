# Phantom-FHE CKKS 接口演示

本目录是一个**独立**的最小示例工程，演示如何用 [Phantom-FHE](https://github.com/encryptorion-lab/phantom-fhe)
（CUDA 加速的全同态加密库）完成 CKKS 方案下的：

1. **加密 / 解密**
2. **同态加法 / 乘法**
3. **密文矩阵 × 向量乘法**（对角线编码 + 旋转法，Halevi–Shoup）

源码：`ckks_demo.cu`（全部接口用法都在里面，带中文注释）。

---

## 为什么选 Phantom-FHE

- **轻量、易读**：单一 C++/CUDA 库，无重型第三方依赖（不像 FIDESlib 需要打补丁的 OpenFHE）。
- **关键模块已用 CUDA 优化**：NTT / 多项式乘法 / key-switching 等核心算子都是原生 CUDA kernel，便于阅读和二次扩展。
- **支持 BGV / BFV / CKKS**（不含 bootstrapping），适合 PPML / 密文推理类应用。

---

## 目录关系

```
FHE/CPU/
├── phantom-fhe/        # clone 下来的 Phantom 库（保持原样，未修改源码）
│   ├── build-wsl/      # 库在 WSL 下的独立编译产物
│   └── setup_env.bat   # Windows 原生编译用的环境脚本（见下）
└── ckks-demo/          # ← 本示例工程，独立于 phantom-fhe
    ├── ckks_demo.cu
    ├── CMakeLists.txt  # 通过 add_subdirectory 引用 ../phantom-fhe
    └── README.md
```

`CMakeLists.txt` 通过 `add_subdirectory(../phantom-fhe)` 引用库源码，**不需要先安装 Phantom**，
配置时会自动连同库一起编译。

---

## 环境要求

本机已配置好的工具链（仅供参考，换机请对应调整）：

| 组件 | 版本 / 位置 |
|------|-------------|
| GPU | NVIDIA RTX 4060（计算能力 `sm_89`）|
| CUDA Toolkit | 12.6 |
| 编译环境 | **WSL2 Ubuntu-22.04**（推荐）：gcc 11.4 + nvcc 12.6 |

> **重要：在 Windows 原生 MSVC 下无法直接编译 Phantom。**
> Phantom 源码使用了 `unsigned __int128`（GCC/Clang 扩展），MSVC 不支持，编译会报错。
> 因此本项目在 **WSL2（Linux）** 下编译，gcc 原生支持 `__int128`，且 RTX 4060 在 WSL2 中可直接访问。
> WSL 内 CUDA 路径：`/usr/local/cuda-12.6`。

---

## 编译与运行（WSL2）

在 PowerShell 里一条命令进入 WSL 完成配置 + 编译 + 运行：

```powershell
wsl --cd "/mnt/d/LEARN/CKKS-MNIST/FHE/CPU/ckks-demo" -e bash -lc '
  export PATH="/usr/local/cuda-12.6/bin:$PATH";
  export CUDACXX=/usr/local/cuda-12.6/bin/nvcc;
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 &&
  cmake --build build -j$(nproc) &&
  cd build &&
  export LD_LIBRARY_PATH="$PWD/phantom-build/lib:/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH" &&
  ./ckks_demo
'
```

也可以先进入 WSL 再分步执行：

```bash
# 进入 WSL
wsl
cd /mnt/d/LEARN/CKKS-MNIST/FHE/CPU/ckks-demo
export PATH="/usr/local/cuda-12.6/bin:$PATH"

# 配置 + 编译
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j$(nproc)

# 运行（需要把库目录加入动态库搜索路径）
cd build
export LD_LIBRARY_PATH="$PWD/phantom-build/lib:/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH"
./ckks_demo
```

> 换 GPU 时把 `-DCMAKE_CUDA_ARCHITECTURES=89` 改成对应计算能力（或用 `native` 自动探测）。

### 预期输出

```
========== [1] 加密 / 解密 ==========
原始明文 : [1.0000, 2.0000, 3.0000, 4.0000, ...]
解密结果 : [1.0000, 2.0000, 3.0000, 4.0000, ...]

========== [2] 同态加法 / 乘法 ==========
a + b    : [3.0000, 4.0000, ...]
a * b    : [2.0000, 4.0000, ...]

========== [3] 密文矩阵 x 向量 (y = M*x) ==========
密文计算 y: [30.0000, 6.0000, 8.0000, 10.0000]
明文期望 y: [30.0000, 6.0000, 8.0000, 10.0000]
最大误差 = ~1e-5
```

---

## 接口速查（CKKS）

所有运算都围绕一个 `PhantomContext`，对象之间的关系：

```
EncryptionParameters ──► PhantomContext
                              │
        ┌─────────────────────┼───────────────────────────┐
   PhantomSecretKey      PhantomCKKSEncoder         （运算函数）
        │                     │ encode/decode        add_inplace
   ┌────┼─────┐               │                      multiply / multiply_plain_inplace
 PublicKey RelinKey GaloisKey                         relinearize_inplace
   │       │       │                                  rescale_to_next_inplace
 加密   乘后重线性  旋转                                rotate_inplace
```

### 1. 设置参数与上下文

```cpp
EncryptionParameters parms(scheme_type::ckks);
parms.set_poly_modulus_degree(1 << 14);                 // N，越大越安全也越慢
parms.set_coeff_modulus(CoeffModulus::Create(
        1 << 14, {60, 40, 40, 40, 60}));                // 模数链；中间项数 ≈ 可用乘法深度
parms.set_special_modulus_size(1);                      // hybrid key-switching 的特殊模数个数
PhantomContext context(parms);

double scale = pow(2.0, 40);                            // 与中间模数位宽匹配
```

- **`poly_modulus_degree (N)`**：slot 数 = N/2。决定安全性与性能。
- **`coeff_modulus`**：第一个和最后一个取 60 bit，中间取 40 bit；中间 40-bit 模数的个数 ≈ 允许的乘法/rescale 次数。
- **`scale`**：建议接近中间模数位宽（这里 2^40）。

### 2. 密钥生成

```cpp
PhantomSecretKey  secret_key(context);                       // 私钥
PhantomPublicKey  public_key = secret_key.gen_publickey(context);
PhantomRelinKey   relin_keys = secret_key.gen_relinkey(context);   // 密文*密文后用
PhantomGaloisKey  galois_keys = secret_key.create_galois_keys(context); // 旋转用
PhantomCKKSEncoder encoder(context);
```

### 3. 编码 / 加密 / 解密 / 解码

```cpp
std::vector<double> data(encoder.slot_count(), 0.0);
data[0] = 3.14;

PhantomPlaintext plain;
encoder.encode(context, data, scale, plain);                 // 向量 -> 明文

PhantomCiphertext cipher;
public_key.encrypt_asymmetric(context, plain, cipher);       // 公钥加密
// 或：secret_key.encrypt_symmetric(context, plain, cipher); // 私钥对称加密

PhantomPlaintext out;
secret_key.decrypt(context, cipher, out);                    // 解密

std::vector<double> result;
encoder.decode(context, out, result);                        // 明文 -> 向量
```

### 4. 同态运算

```cpp
add_inplace(context, c1, c2);                  // c1 += c2

multiply_plain_inplace(context, c, plain);     // 密文 × 明文
rescale_to_next_inplace(context, c);           // 乘后 rescale

PhantomCiphertext c3 = multiply(context, c1, c2);  // 密文 × 密文
relinearize_inplace(context, c3, relin_keys);      // 3 项 -> 2 项
rescale_to_next_inplace(context, c3);              // rescale

rotate_inplace(context, c, k, galois_keys);    // slot 循环左移 k 位
```

> **乘法的两条铁律**：
> 1. 密文×密文后必须 `relinearize_inplace`（恢复成 2 项密文）。
> 2. 任何乘法后都要 `rescale_to_next_inplace` 控制 scale 增长；rescale 次数受模数链长度限制（即乘法深度有限）。

### 5. 矩阵 × 向量（对角线 + 旋转法）

计算 `y = M·x`（M 为 d×d 明文矩阵，x 为加密向量）：

```
y = Σ_{k=0}^{d-1}  diag_k(M) ⊙ rot(x, k)
其中 diag_k(M)[i] = M[i][(i+k) mod d]
```

只需 **d 次明文乘 + (d-1) 次旋转**，全程在密文上完成。详见 `ckks_demo.cu` 中的 `demo_matrix_vector`。

> **坑点**：`rotate_inplace` 是对**整个 slot 空间**做循环旋转，不是对长度为 d 的子块。
> 所以要把向量 `x` 和对角线 `diag` **以周期 d 平铺**填满所有 slot，
> 这样全局旋转在每个 d-块内才等价于块内循环旋转（示例已处理）。

---

## 后续扩展建议（面向 FHE-MNIST）

- 把矩阵乘法封装成 `Linear` 层（权重明文、激活密文），用线性/多项式近似替代 ReLU。
- 多层网络注意乘法深度：每层一次乘法消耗一个模数，预留足够的 `coeff_modulus` 项。
- 性能热点都在 Phantom 的 CUDA kernel（`phantom-fhe/src/ntt/`、`evaluate.cu` 等），需要进一步加速时从这里入手。
