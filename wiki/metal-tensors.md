# TensorOps、量化与融合候选

[English companion](en/metal-tensors.md)

适用：prefill GEMM、量化线性层和自定义 attention。**本文是未执行的优化方案；没有本地速度提升声明。**

## 先决定是否值得下到 kernel

先分析目标框架实际热点。大 GEMM 可尝试 MPP `mpp::tensor_ops`；小 batch decode 则先核查权重/KV 读取和调度开销。M5 专用矩阵硬件不能保证小矩阵或端到端同幅提速。M5 技术讲座给出的版本节点是：26.1 BF16 tensor、26.3 cooperative tensor 输入 matmul、26.4 INT4/INT8 tensor；必须继续核查具体 symbol 的 SDK availability。[M5 ML 讲座](https://developer.apple.com/videos/play/tech-talks/111432/)

旧式8×8 `simdgroup_matrix` 与 MPP cooperative tensor 的区别，见 [MSL 矩阵乘](msl-matrix.md)。该页的 M4 运行只验证公开 SIMD-group 示例，没有运行本页 MPP/TensorOps 候选。

## GEMM 先验证 cache 与 tile

MPP 指南的 GEMM 路径强调直接访问 device memory 并利用 cache；不要默认移植 CUDA 的 shared-memory staging 和软件流水线。候选变量包括 SIMD/threadgroup tile、保持局部性的遍历顺序、完整 tile 的静态 extent、K 分段频率、寄存器内 epilogue。指南的 M5 起始参数只是该目标的候选，不能跨代固定。[MPP 指南](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf)

危险点：大 tile 会减少并行度并增加 private-memory 压力；静态 extent 不能用于越界尾块；tensor operation 指定的作用域内线程必须共同参与。用于协调进度的 `mem_none` barrier 不承担生产者/消费者所需的内存排序。不要把该 GEMM 指南解读成所有算法都禁止线程组内存。

## 量化和 attention 的版本边界

WWDC26 增加 FP4/FP8/INT2 与 E8M0 scales plane 的 27 系统路线，并展示 row reduction、cooperative tensor 兼容性检查与中间值复用。若 native quantized tensor 符合模型格式，优先将它作为候选；自定义格式再比较寄存器内反量化和线程组暂存。[WWDC26 TensorOps](https://developer.apple.com/videos/play/wwdc2026/330/)

不要混淆“26.3 起已有 cooperative 输入”与“27 讲座中的新复用辅助接口”。具体方法仍由 SDK 声明决定。不同 cooperative tensor 未必兼容；不能直接复用时应明确转换。新量化格式还要求正确的 stride、offset、block scale 和 alignment，不能只改 dtype 名称。

## 验证方案（未执行）

1. 固定 shape、dtype、量化 scheme、目标与正确性容差，保留框架 baseline。
2. 分别比较 tile/cache、native/custom dequant、epilogue 或 attention fusion，避免一次改变全部变量。
3. 检查尾块、转置/stride、mask、全 mask 行、累加精度及量化质量；测 kernel 与端到端性能。
4. 用 profiler 确认瓶颈及硬件路径变化；只有编译成功不能证明使用专用矩阵单元。

正确但变慢的 shape 不应被宽泛 dispatch 掩盖；保留显式适用范围与原始基线。

量化候选还需要区分存储收益与计算收益：减小模型体积不保证反量化后算子更快，算子误差合格也不保证生成质量保持。记录权重位宽、分组大小、缩放方式、累加类型与模型评测结果。若只测过一个矩阵尺寸，应将条目标为该尺寸的候选证据，等待边界与真实请求验证后再扩大适用范围。
