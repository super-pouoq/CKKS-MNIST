# CKKS-MNIST 后端 (Flask)

接收前端上传的图片，归一化成 MNIST 输入后调用已编译好的 CUDA 程序
`../GPU/ckks-mnist/build/ckks_mnist` 做全密文 CKKS 推理，并把结果返回前端。

## 端口

- 监听 `0.0.0.0:5000`

## 接口

### `GET /api/health`

健康检查。

返回：

```json
{ "status": "ok", "ckks_dir": "/mnt/d/LEARN/CKKS-MNIST/FHE/GPU/ckks-mnist" }
```

### `POST /api/predict`

对一张图片做加密推理。请求体 JSON（二选一）：

| 字段           | 类型     | 说明                                              |
| -------------- | -------- | ------------------------------------------------- |
| `image_base64` | string   | 图片的 base64（可带 `data:image/png;base64,` 前缀）|
| `pixels`       | number[] | 长度 784 的 0–255 灰度数组（28×28 展开）           |
| `invert`       | bool     | 可选，白底黑字时设为 `true` 做反相，默认 `false`   |

后端会把图片转灰度、缩放到 28×28，并按训练时的归一化处理
（mean=0.1307, std=0.3081），写成 `images.txt` 后调用 CUDA 程序。

返回：

```json
{
  "prediction": 7,
  "logits": [-14.11, -5.43, -9.81, -6.83, -9.66, -9.52, -14.56, 28.97, -9.49, -5.13],
  "slot_count": 16384,
  "mid_levels": 9,
  "raw_stdout": "..."
}
```

出错时返回 `4xx/5xx` 与 `{ "error": "..." }`。
