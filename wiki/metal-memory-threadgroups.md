# Metal 内存、线程组与同步

适用：GEMV、RMSNorm、RoPE、softmax、小矩阵乘，及频繁读回或分配导致的慢推理。**以下为文档支持的候选方法，尚未执行设备验证。**

## 线程组从 pipeline 出发

查询 `threadExecutionWidth`、`maxTotalThreadsPerThreadgroup` 和静态线程组内存。最大线程数与 kernel 使用的寄存器、内存有关，同一设备的两个 pipeline 也可能不同。候选搜索可比较若干 execution-width 倍数，同时改变每线程工作量；最大合法线程组不保证最快。[线程组与网格计算](https://developer.apple.com/documentation/metal/calculating-threadgroup-and-grid-sizes)

不要把 SIMD 宽度写成所有 Metal 设备通用的常量。支持 nonuniform dispatch 也不代表任何 collective kernel 都能接受不完整尾组：矩阵、归约和 barrier 必须满足参与线程的约定。非整除形状需要安全的尾部实现。[Apple Silicon 移植说明](https://developer.apple.com/documentation/apple-silicon/porting-your-metal-code-to-apple-silicon)

## 统一内存仍然有访问时序

CPU 填充、GPU 读取的资源先考虑 shared；GPU 生成和消费的中间值考虑 private。shared 允许双方访问同一系统内存，却不允许 CPU 在 GPU 未完成时读写同一数据。private 在 Apple Silicon 上不意味着独立显存。memoryless 只适用于特定临时纹理，不是存放 LLM buffer 的模式。[存储模式](https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus)；[shared 访问要求](https://developer.apple.com/documentation/metal/mtlstoragemode/shared)

候选：复用 buffer，保留 GPU 中间结果，减少读回和格式转换。可以为独立请求轮换资源实例以重叠 CPU/GPU 工作；不要把渲染的多缓冲机械套到有逐 token 依赖的 decode。[CPU/GPU 同步示例](https://developer.apple.com/documentation/metal/synchronizing-cpu-and-gpu-work)

## 同步与占用率的决策

- SIMD reduction 不能代替共享内存的排序。即使线程组只有一个 SIMD group，存在相关读写时仍需适当 barrier；所有参与线程必须到达正确的同步点。
- occupancy 低先检查 ALU/带宽是否已饱和；若未饱和，再区分网格过小、线程组内存、live registers、cache/MMU 压力。不要只追求百分比。[Apple9 profiling](https://developer.apple.com/videos/play/tech-talks/111374/)
- 大 tile 或过度 fusion 可能增加 live registers 和 stack spill。M5 工具的 occupancy influence 与逐行寄存器信息可指导缩短生命周期、减小 tile 或拆分 fusion；减少寄存器本身不是性能结论。[M5 profiling](https://developer.apple.com/videos/play/tech-talks/111431/)

## 验证方案（未执行）

在相同输入与环境下比较小/大线程组、复用 buffer 与现有实现；覆盖非整除尾部、极小/极大行、mask 和不同数值分布。保留输出误差、kernel 时间、端到端 token 延迟、分配/复制/等待及可用的 occupancy/spill 证据。缺少某类 counter 必须写明覆盖限制。

归约改变求和顺序时，即使内存访问完全合法，浮点结果仍可能变化。先确定参考实现与容差，再检查极值、接近零的输入和长行累加；不要用一次随机输入通过替代数值覆盖。同步候选出现偶发错误时，先恢复正确时序，再讨论性能。
