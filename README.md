# CKKS-MNIST
## TODO LIST


## 1. 明文模型

- [x] 训练 FHE-friendly MNIST 模型
- [x] 用线性函数替代 ReLU
- [x] 测试明文准确率
- [x] 保存模型权重
- [x] 导出推理所需参数
- [x] 记录明文推理耗时

---

## 2. FHE 推理流程

- [x] 生成 CKKS 参数与密钥
- [x] 实现图片编码与加密
- [x] 实现密文推理流程
- [x] 实现结果解密与分类
- [x] 跑通单张 MNIST FHE 推理
- [x] 统计 FHE 准确率与耗时

---

## 3. Phantom GPU 实验

- [x] 跑通 GPU 端 CKKS 基础操作
- [ ] 对比 OpenFHE CPU 与 Phantom GPU 性能
- [ ] 记录 Add / Mul / Square / Rotate 等操作耗时
- [ ] 统计 GPU 加速比
- [ ] 整理 primitive benchmark 图表

---

## 4. 参考 HEngine 的优化

- [ ] 阅读并整理 HEngine 关键优化点
- [ ] 选择一个主要优化方向
- [ ] 优先考虑 BSGS + Hoisted Rotation
- [ ] 可选考虑 Lazy Mask
- [ ] 可选考虑 batch / stream 摊销优化
- [ ] 对比 baseline 与优化后性能
- [ ] 整理操作次数、rotation 次数、耗时对比

---

## 5. Kernel Fusion / Pipeline 优化

- [ ] 分析 FIDESlib 推理中的数据流
- [ ] 分析 OpenFHE 与 FIDESlib 之间的转换开销
- [ ] 尝试减少中间密文的 CPU-GPU 往返
- [ ] 尝试让连续密文操作尽量保留在 GPU 侧
- [ ] 对比优化前后的总推理耗时
- [ ] 整理 pipeline-level fusion 实验结果

---

## 6. 软件展示

- [ ] 搭建展示界面
- [x] 支持上传或选择 MNIST 图片
- [ ] 展示加密前图片
- [ ] 展示密文推理流程
- [ ] 展示解密预测结果
- [ ] 展示耗时统计
- [ ] 展示 GPU 加速实验图表
- [ ] 准备固定演示样例

---

## 7. 实验结果整理

- [ ] 明文 vs FHE 准确率
- [ ] 明文 vs FHE 推理耗时
- [ ] CPU CKKS vs GPU CKKS primitive 耗时
- [ ] baseline vs optimized encrypted linear
- [ ] batch size vs amortized latency
- [ ] pipeline 优化前后对比
- [ ] 生成最终表格和图

---

## 8. 文档与答辩

- [ ] 写 README
- [ ] 写系统设计文档
- [ ] 写威胁模型说明
- [ ] 写 GPU 加速说明
- [ ] 写 HEngine 参考与优化说明
- [ ] 写实验结果分析
- [ ] 制作 PPT
- [ ] 录制演示视频
- [ ] 准备备用演示方案
- [ ] 完成最终彩排

---

## 9. 最低交付目标

- [ ] FHE-MNIST 能演示
- [ ] FIDESlib GPU benchmark 能运行
- [ ] 至少有一个参考 HEngine 的优化实验
- [ ] 有 CPU/GPU 对比数据
- [ ] 有完整 PPT 和作品说明
- [ ] 有演示视频作为备份

---

## 10. 冲刺目标

- [ ] 完整 FIDESlib-based 密文推理
- [ ] BSGS-Hoisted encrypted linear 优化
- [ ] Lazy Mask 优化
- [ ] pipeline-level fusion 优化
- [ ] 多 batch 摊销推理实验