#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CKKS-MNIST 演示后端 (Flask)。

职责:
  1. 接收前端上传的图像 (PNG/JPG/base64 或 28x28 像素数组)。
  2. 把图像处理成 MNIST 输入: 灰度 -> 28x28 -> 归一化 (mean=0.1307, std=0.3081)。
  3. 写成 ckks_mnist 约定的 images.txt 文本格式。
  4. 在 WSL 中调用已编译好的 CUDA 程序 build/ckks_mnist 做全密文 CKKS 推理。
  5. 解析其 stdout, 把 label / pred / 10 个 logit 返回给前端。
"""

import os
import io
import re
import base64
import subprocess
import tempfile

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


# --------------------------- 路径推导 ---------------------------
def win_to_wsl(path: str) -> str:
    """把 Windows 绝对路径转成 WSL 路径: D:\\foo\\bar -> /mnt/d/foo/bar。"""
    path = os.path.abspath(path).replace("\\", "/")
    if len(path) > 1 and path[1] == ":":
        path = "/mnt/" + path[0].lower() + path[2:]
    return path


# backend/app.py -> 上一级是 FHE/, 其下 GPU/ckks-mnist 即 CUDA 工程。
# 全部从本文件位置推导, 不写死绝对路径; 需要时仍可用环境变量覆盖。
_HERE = os.path.dirname(os.path.abspath(__file__))
CKKS_DIR_WIN = os.environ.get(
    "CKKS_DIR_WIN",
    os.path.normpath(os.path.join(_HERE, "..", "GPU", "ckks-mnist")))
CKKS_DIR_WSL = os.environ.get("CKKS_DIR_WSL", win_to_wsl(CKKS_DIR_WIN))
CUDA_DIR = os.environ.get("CUDA_DIR", "/usr/local/cuda-12.6")

app = Flask(__name__)
CORS(app)


# --------------------------- 图像处理 ---------------------------
def image_to_norm_28x28(img: Image.Image, invert: bool = False) -> np.ndarray:
    """灰度化 + 缩放到 28x28 + 与训练一致的归一化, 返回长度 784 的 float 向量。"""
    img = img.convert("L").resize((28, 28), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if invert:  # 画板/白底黑字时需要反相, 使笔画为高亮 (与 MNIST 一致)
        arr = 1.0 - arr
    arr = (arr - MNIST_MEAN) / MNIST_STD
    return arr.reshape(-1)


def parse_payload(payload: dict):
    """支持两种输入: base64 图片 / 直接的 28x28(或 784) 像素数组。"""
    invert = bool(payload.get("invert", False))

    if "image_base64" in payload and payload["image_base64"]:
        raw = payload["image_base64"]
        if "," in raw:  # 去掉 data:image/png;base64, 前缀
            raw = raw.split(",", 1)[1]
        img = Image.open(io.BytesIO(base64.b64decode(raw)))
        return image_to_norm_28x28(img, invert=invert)

    if "pixels" in payload and payload["pixels"]:
        pixels = np.asarray(payload["pixels"], dtype=np.float32).reshape(-1)
        if pixels.size != 784:
            raise ValueError(f"pixels 长度应为 784, 实际 {pixels.size}")
        # 约定: pixels 为 0..255 灰度
        arr = pixels / 255.0
        if invert:
            arr = 1.0 - arr
        return (arr - MNIST_MEAN) / MNIST_STD

    raise ValueError("缺少 image_base64 或 pixels 字段")


# --------------------------- 调用 .cu 推理 ---------------------------
def write_images_txt(norm_vec: np.ndarray, label: int = 0) -> str:
    """写成 ckks_mnist 的 images.txt 格式 (count rows cols / label / pixels)。"""
    tmp_dir = tempfile.mkdtemp(prefix="ckks_req_")
    path = os.path.join(tmp_dir, "images.txt")
    with open(path, "w") as f:
        f.write("1 28 28\n")
        f.write(f"{label}\n")
        f.write(" ".join(repr(float(v)) for v in norm_vec) + "\n")
    return tmp_dir


def run_inference(req_data_dir_win: str):
    """在 WSL 中执行 ckks_mnist, data 目录用临时请求目录(含 images.txt)+ 工程 weights.txt。"""
    req_data_wsl = win_to_wsl(req_data_dir_win)
    # weights.txt 体积大, 复用工程 data 目录里的; images.txt 用本次请求生成的。
    bash = (
        f'export PATH="{CUDA_DIR}/bin:$PATH"; '
        f'cd "{CKKS_DIR_WSL}/build"; '
        f'export LD_LIBRARY_PATH="$PWD/phantom-build/lib:{CUDA_DIR}/lib64:$LD_LIBRARY_PATH"; '
        f'cp "{CKKS_DIR_WSL}/data/weights.txt" "{req_data_wsl}/weights.txt"; '
        f'./ckks_mnist "{req_data_wsl}" 1'
    )
    proc = subprocess.run(
        ["wsl", "-e", "bash", "-lc", bash],
        capture_output=True, text=True, timeout=180)
    return proc


def parse_output(stdout: str):
    """解析形如: img0: label=7 pred=7  logits=[..10..]。"""
    m = re.search(r"pred=(\d+)\s+logits=\[([^\]]+)\]", stdout)
    if not m:
        return None
    pred = int(m.group(1))
    logits = [float(x) for x in m.group(2).split()]
    slot = re.search(r"slot_count=(\d+)", stdout)
    levels = re.search(r"mid levels=(\d+)", stdout)
    return {
        "prediction": pred,
        "logits": logits,
        "slot_count": int(slot.group(1)) if slot else None,
        "mid_levels": int(levels.group(1)) if levels else None,
    }


# --------------------------- 路由 ---------------------------
@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "ckks_dir": CKKS_DIR_WSL})


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    try:
        norm_vec = parse_payload(payload)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"图像解析失败: {e}"}), 400

    tmp_dir = write_images_txt(norm_vec)
    try:
        proc = run_inference(tmp_dir)
    except subprocess.TimeoutExpired:
        return jsonify({"error": "推理超时 (>180s)"}), 504
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"调用 ckks_mnist 失败: {e}"}), 500

    if proc.returncode != 0:
        return jsonify({
            "error": "ckks_mnist 非零退出",
            "stderr": proc.stderr[-2000:],
            "stdout": proc.stdout[-2000:],
        }), 500

    result = parse_output(proc.stdout)
    if result is None:
        return jsonify({
            "error": "无法解析推理输出",
            "stdout": proc.stdout[-2000:],
        }), 500

    result["raw_stdout"] = proc.stdout
    return jsonify(result)


if __name__ == "__main__":
    print(f"[ckks-mnist] CKKS_DIR_WIN = {CKKS_DIR_WIN}")
    print(f"[ckks-mnist] CKKS_DIR_WSL = {CKKS_DIR_WSL}")
    app.run(host="0.0.0.0", port=5000, debug=True)
