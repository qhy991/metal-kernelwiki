# Metal LLM 性能诊断与证据

[English companion](en/metal-profiling.md)

适用：首 token 慢、decode 吞吐低、GPU 间歇空闲、融合 kernel 变慢。**以下采集与比较方案未执行。**

## 先看整个请求，再看 kernel

Metal System Trace 用于定位 CPU 编码、GPU 执行、内存活动和等待；Xcode Metal debugger 用于捕获后的资源、pipeline、shader 与 counter 检查。当前 Apple 工具页还列出 `gpucapture`、`gpudebug`、`metalperftrace`。它们是可选入口，先核查安装的 Xcode 与工具帮助，不向旧环境承诺存在新命令或 counter。[Metal 工具](https://developer.apple.com/metal/tools/)

```text
系统 trace → 关键时间段 → 实际 pipeline → limiter/counter
→ 一个可证伪假设 → 正确性检查 → 不带 capture 的基准
```

采集 prefill 与 decode 各自代表性片段，记录 prompt/batch/context/token 数、权重/KV dtype、缓存状态和计时范围。把模型加载、编译/预热与稳定推理分开说明；不要把 CPU enqueue 时间当 GPU 完成时间。TTFT 包含哪些阶段必须明说。

## 从观测选择下一步

- **GPU 空闲、CPU 忙**：检查编码/提交和同步；尝试合并兼容 dispatch、复用对象，防止批次太大增加排队延迟。[Command buffer 权衡](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/CommandBuffers.html)
- **occupancy 低、ALU 已满**：先优化 ALU 工作，不预设增加 occupancy 会改善。
- **occupancy 与 ALU 都低**：检查 launch、线程组内存、occupancy target，再分析 L1/LLC/MMU；counter 名称与可用性依目标而定。[Apple9 工具讲座](https://developer.apple.com/videos/play/tech-talks/111374/)
- **fusion 后退化**：检查 live registers、stack spill 与缓存压力，再比较较小 tile 或拆分方案。[M5 工具讲座](https://developer.apple.com/videos/play/tech-talks/111431/)

Metal 4 的低开销 queue 模型也改变了同步责任。有资源访问冲突时需要正确 barrier/fence/event；不可照搬 Metal 3 的隐式 hazard 假设。[Metal 4 核心 API](https://developer.apple.com/documentation/metal/understanding-the-metal-4-core-api)；[资源同步](https://developer.apple.com/documentation/metal/resource-synchronization)

## 证据记录（未执行模板）

```text
target/toolchain/backend:
workload and timing boundary:
reference + tolerance + correctness:
candidate + expected counter change:
baseline/candidate samples and statistic:
trace/capture/log paths:
observed bottleneck / missing coverage:
end-to-end result / next hypothesis:
```

使用 counter sample buffer 前枚举支持的 counter set 和采样位置；GPU/CPU 时间线对齐可能需要时钟换算。[GPU counter API](https://developer.apple.com/documentation/metal/gpu-counters-and-counter-sample-buffers)

profiler 用于解释，最终延迟由控制好预热、样本和同步边界的基准决定。报告缺失的 counter、未测形状和未验证设备；缺少观测不能写成“没有瓶颈”。保留原始结果，失败诊断不因后续候选成功而覆盖。

重复测量时固定电源模式、并发任务、请求长度和缓存策略，并记录可观察到的温度或降频状态。分别报告单请求延迟与并发吞吐，不能用独立调用的高吞吐替代有依赖链的逐步生成延迟。诊断记录应包含支持假设的观测和反证：例如复制减少但总延迟未变，意味着下一步需要寻找新的关键路径。若采集工具没有目标权限或环境不满足要求，应保留错误并说明前提，不能把修复环境后的一次结果冒充原始状态。
