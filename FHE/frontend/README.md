# CKKS-MNIST 前端 (Vue 3 + Vite)

提供手写画板 / 图片上传界面，把图像发给后端做全密文 CKKS 推理，并展示
预测数字与 10 个 logit。

## 端口

- 开发服务器 `http://localhost:5173`
- `/api` 已配置代理到后端 `http://localhost:5000`，无需关心跨域

## 与后端的交互

页面把画板内容导出为 PNG（base64），调用一个后端接口：

### `POST /api/predict`

请求体：

```json
{ "image_base64": "data:image/png;base64,....", "invert": false }
```

返回（由后端给出）：

```json
{
  "prediction": 7,
  "logits": [/* 10 个数 */],
  "slot_count": 16384,
  "mid_levels": 9
}
```

页面用 `prediction` 显示预测数字，用 `logits` 画出各类别的条形图。
