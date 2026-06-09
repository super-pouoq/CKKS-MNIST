# 运行指南：CKKS-MNIST 前后端演示

前端提供手写/上传界面，后端接收图片、归一化后调用已编译好的 CUDA 程序
（`FHE/GPU/ckks-mnist/build/ckks_mnist`）做全密文 CKKS 推理，再把结果返回前端展示。

## 服务地址

| 服务 | 目录          | 地址                    |
| ---- | ------------- | ----------------------- |
| 前端 | `FHE/frontend` | http://localhost:5173 |
| 后端 | `FHE/backend`  | http://localhost:5000 |

前端的 `/api` 已代理到后端 `:5000`，无需关心跨域。

## 启动后端

```bash
cd FHE/backend
python app.py
```

监听 `http://localhost:5000`。

## 启动前端

```bash
cd FHE/frontend
npm install
npm run dev
```

打开 http://localhost:5173 即可使用。

> 后端会在 WSL 中调用 CUDA 程序，请确保 `FHE/GPU/ckks-mnist/build/ckks_mnist` 已编译存在。
