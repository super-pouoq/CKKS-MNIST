#!/usr/bin/env python3
# export_model.py
# 无需 torch：直接解析 PyTorch .pth(zip) 中的原始 float32 storage，
# 导出权重为简单的文本格式 weights.txt；同时从 MNIST raw idx 文件导出若干
# 测试样本(已做与训练相同的 Normalize)到 images.txt，供 C++ CKKS 推理读取。
#
# 网络结构 (model.py):
#   conv1: Conv2d(1,16,3,pad=1)   weight [16,1,3,3]  bias [16]
#   pool1: AvgPool2d(2)           square activation 在卷积后
#   conv2: Conv2d(16,32,3,pad=1)  weight [32,16,3,3] bias [32]
#   pool2: AvgPool2d(2)
#   fc1:   Linear(32*7*7,128)     weight [128,1568]  bias [128]   square
#   fc2:   Linear(128,10)         weight [10,128]    bias [10]
#
# state_dict 键顺序与 data/N storage 的对应关系由 data.pkl 决定，
# 这里用一个最小的 Unpickler 还原 (key -> (storage_key, dtype, shape))。

import os
import sys
import zipfile
import pickle
import struct
import io

PTH = sys.argv[1] if len(sys.argv) > 1 else \
    "/mnt/d/LEARN/CKKS-MNIST/MNIST/normal/model/fhe_friendly_cnn.pth"
MNIST_RAW = sys.argv[2] if len(sys.argv) > 2 else \
    "/mnt/d/LEARN/CKKS-MNIST/MNIST/normal/data/MNIST/raw"
OUT_DIR = sys.argv[3] if len(sys.argv) > 3 else \
    "/mnt/d/LEARN/CKKS-MNIST/FHE/GPU/ckks-mnist/data"
NUM_IMAGES = int(sys.argv[4]) if len(sys.argv) > 4 else 20

# ---- 与训练一致的归一化常数 ----
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

# ---------------------------------------------------------------------------
# 1) torch-free 解析 .pth
# ---------------------------------------------------------------------------
DTYPE_BYTES = {"float": 4, "double": 8, "long": 8, "int": 4}

class _Storage:
    def __init__(self, key, dtype_str):
        self.key = key
        self.dtype_str = dtype_str

class _Unpickler(pickle.Unpickler):
    """只用于还原 state_dict 的结构, 把 tensor 还原成 (storage_key, dtype, shape, stride)。"""
    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return self._rebuild_tensor_v2
        if module == "torch._utils" and name == "_rebuild_parameter":
            return self._rebuild_parameter
        if module == "collections" and name == "OrderedDict":
            from collections import OrderedDict
            return OrderedDict
        if module == "torch" and name in ("FloatStorage", "DoubleStorage",
                                          "LongStorage", "IntStorage"):
            dmap = {"FloatStorage": "float", "DoubleStorage": "double",
                    "LongStorage": "long", "IntStorage": "int"}
            return lambda *a, **k: dmap[name]
        # 其它一律返回占位
        return lambda *a, **k: None

    def persistent_load(self, pid):
        # pid 形如 ("storage", FloatStorage, key, location, numel)
        typ = pid[0]
        assert typ == "storage"
        dtype_str = pid[1] if isinstance(pid[1], str) else "float"
        key = pid[2]
        return _Storage(key, dtype_str)

    @staticmethod
    def _rebuild_tensor_v2(storage, storage_offset, size, stride,
                           requires_grad=False, backward_hooks=None, *extra):
        return {"storage_key": storage.key, "dtype": storage.dtype_str,
                "offset": storage_offset, "shape": tuple(size),
                "stride": tuple(stride)}

    @staticmethod
    def _rebuild_parameter(data, requires_grad=False, backward_hooks=None):
        return data


def load_state_dict(pth_path):
    z = zipfile.ZipFile(pth_path)
    root = z.namelist()[0].split("/")[0]
    with z.open(f"{root}/data.pkl") as f:
        meta = _Unpickler(io.BytesIO(f.read())).load()
    out = {}
    for k, t in meta.items():
        nbytes = DTYPE_BYTES[t["dtype"]]
        raw = z.read(f"{root}/data/{t['storage_key']}")
        n = 1
        for s in t["shape"]:
            n *= s
        if t["dtype"] == "float":
            vals = list(struct.unpack(f"<{n}f", raw[:n*4]))
        elif t["dtype"] == "double":
            vals = list(struct.unpack(f"<{n}d", raw[:n*8]))
        else:
            raise RuntimeError("unexpected dtype " + t["dtype"])
        out[k] = (t["shape"], vals)
    return out


# ---------------------------------------------------------------------------
# 2) 读取 MNIST raw idx
# ---------------------------------------------------------------------------
def read_idx_images(path):
    with open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051
        data = f.read(num * rows * cols)
    return num, rows, cols, data

def read_idx_labels(path):
    with open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        assert magic == 2049
        data = f.read(num)
    return list(data)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sd = load_state_dict(PTH)
    print("loaded state_dict keys:")
    for k, (shape, _) in sd.items():
        print(f"  {k}: {shape}")

    # 写权重: 每个张量一行头 "name d0 d1 ..." 再一行所有数值(空格分隔)
    wpath = os.path.join(OUT_DIR, "weights.txt")
    order = ["conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias",
             "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"]
    with open(wpath, "w") as f:
        for name in order:
            shape, vals = sd[name]
            f.write(name + " " + " ".join(str(s) for s in shape) + "\n")
            f.write(" ".join(repr(v) for v in vals) + "\n")
    print("wrote", wpath)

    # 写测试图片: 第一行 "count rows cols"; 之后每张图: "label" 行 + 像素行(已归一化)
    num, rows, cols, idata = read_idx_images(os.path.join(MNIST_RAW, "t10k-images-idx3-ubyte"))
    labels = read_idx_labels(os.path.join(MNIST_RAW, "t10k-labels-idx1-ubyte"))
    k = min(NUM_IMAGES, num)
    ipath = os.path.join(OUT_DIR, "images.txt")
    with open(ipath, "w") as f:
        f.write(f"{k} {rows} {cols}\n")
        for i in range(k):
            base = i * rows * cols
            px = idata[base: base + rows * cols]
            norm = [((b / 255.0) - MNIST_MEAN) / MNIST_STD for b in px]
            f.write(str(labels[i]) + "\n")
            f.write(" ".join(repr(v) for v in norm) + "\n")
    print(f"wrote {ipath} ({k} images, {rows}x{cols})")


if __name__ == "__main__":
    main()
