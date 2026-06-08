# CKKS-MNIST：Phantom-FHE 上的 MNIST 加密推理

本目录在 **不接触明文数据** 的前提下，用 [Phantom-FHE](https://github.com/encryptorion-lab/phantom-fhe)（CUDA 加速的 CKKS）
对已训练好的 MNIST 网络做**全密文前向推理**：图片加密后，卷积 / 平方激活 / 平均池化 / 全连接
全部在密文上完成，最后只解密 10 个 logit 取 argmax。

参考工程：同级 `../ckks-demo`（接口用法），本工程独立、互不影响。

---

## 明文模型

权重来自 `../../MNIST/normal/model/fhe_friendly_cnn.pth`（已训练），结构见 `../../MNIST/normal/model.py`：

```
conv1 Conv2d(1,16,3,pad=1) -> x^2 -> AvgPool2d(2)   => 16 x 14 x 14
conv2 Conv2d(16,32,3,pad=1)-> x^2 -> AvgPool2d(2)   => 32 x 7  x 7
flatten(1568) -> fc1 Linear(1568,128) -> x^2 -> fc2 Linear(128,10)
```

该网络已经是 **FHE 友好** 的：用 `x^2` 替代 ReLU、`AvgPool` 替代 `MaxPool`，整网都是多项式。

---

## 目录结构

```
ckks-mnist/
├── export_model.py   # 无需 torch，解析 .pth + MNIST 原始数据，导出 data/weights.txt、data/images.txt
├── verify_plain.py   # 纯 Python 复现前向，校验导出的权重/图片是否正确
├── ckks_mnist.cu     # 主程序：CKKS 加密推理（带中文注释）
├── CMakeLists.txt    # add_subdirectory(../phantom-fhe)
├── data/             # 导出的权重与测试样本（运行 export_model.py 生成）
└── README.md
```

---

## 第一步：导出权重与测试图片

`export_model.py` **不依赖 torch**：直接解包 `.pth`（zip）里的原始 float32 storage，
并把 MNIST 测试图片做与训练相同的归一化（mean=0.1307, std=0.3081）。

```bash
# 默认从 ../../MNIST/normal 读取，导出到 ./data，导出前 20 张测试图
python3 export_model.py
# 自定义： python3 export_model.py <pth> <mnist_raw_dir> <out_dir> <num_images>
```

可选校验（纯 Python 前向，应在 20 张里得到 19/20）：

```bash
python3 verify_plain.py
```

---

## 第二步：编译与运行（WSL2 + CUDA）

> 与 `../ckks-demo` 相同：Phantom 用了 GCC 的 `unsigned __int128`，**MSVC 无法编译**，
> 因此在 WSL2(Linux) 下用 gcc + nvcc 编译。本机 GPU = RTX 4060(sm_89)，CUDA 12.6。

一条命令完成配置 + 编译 + 运行：

```powershell
wsl --cd "/mnt/d/LEARN/CKKS-MNIST/FHE/GPU/ckks-mnist" -e bash -lc '
  export PATH="/usr/local/cuda-12.6/bin:$PATH";
  export CUDACXX=/usr/local/cuda-12.6/bin/nvcc;
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 &&
  cmake --build build -j$(nproc) &&
  cd build &&
  export LD_LIBRARY_PATH="$PWD/phantom-build/lib:/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH" &&
  ./ckks_mnist ../data 20
'
```

`./ckks_mnist <data_dir> <num_images>`：对前 `num_images` 张做加密推理。

### 预期输出（节选）

```
slot_count=16384, mid levels=9
img0: label=7 pred=7  logits=[... 28.97 ...]
img1: label=2 pred=2  logits=[... 52.10 ...]
...
encrypted acc: 19/20
```

密文 logit 与明文前向逐张吻合（误差 ~1e-2），20 张里 19 张正确（与明文模型同一张出错）。

---

## CKKS 实现要点

打包采用 **channel-packing**：每个特征图通道 = 一个密文，像素 `(y,x)` 放在 slot
`y*step_row + x*step_col`。

- **卷积**：对 3×3 卷积核的 9 个 tap，做密文平移 `rotate` + 明文权重乘 + 累加；
  边界用 valid-mask 折进明文系数；偏置最后以明文相加。
- **平均池化**：只用 `rotate + add` 求 2×2 之和（不乘 `1/4`，不消耗乘法层）；
  有效值保留在 `(2oy,2ox)` 的 **stride 布局**上（布局步长翻倍），`1/4` 折叠进下一层权重。
- **全连接**：明文权重铺到各通道的有效 slot，`multiply_plain` 后用折半 `rotate-sum`
  收缩求和，再用 one-hot 掩码把第 `o` 个结果落到 slot `o`。

### 乘法深度与参数

关键路径的 rescale 次数：

```
conv1(1) | square(1) | conv2(1) | square(1) | fc1(乘1+掩码1=2) | square(1) | fc2(2) = 9
```

因此取 `N = 2^15`、模数链 `{60, 40×9, 60}`（9 个中间 40-bit 级），`scale = 2^40`，
`special_modulus_size = 1`。该参数在 128-bit 安全下（`N=2^15` 上限约 881 bit）成立。

> 想更快可改用 `N=2^14`，但需进一步压低乘法深度（例如把池化求和折进 fc1、合并掩码层），
> 否则模数链不足。当前实现以**正确性优先**。
