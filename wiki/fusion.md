# 融合与提交：先证明开销在哪里

当 trace 显示 GPU 间隙、许多短 dispatch、CPU 编码/采样或中间数组读写时使用本页。下面都是待测候选，不是“kernel 数越少越快”的规则。

## 三个层次分别优化

1. **表达式层**：对稳定的纯子图比较 `mx.compile`，例如相邻 elementwise 激活/缩放。避免在 token 循环中重新创建函数；变 shape/dtype 或错误捕获可变状态会改变行为。详见 [MLX 执行](mlx-execution.md) 和 [编译文档](https://ml-explore.github.io/mlx/build/html/usage/compile.html)。
2. **Kernel 层**：对已有热点考虑 RMSNorm+残差、RoPE/缓存写入、GEMM epilogue 等融合候选。它们必须保持原数据依赖和 dtype；若 live values 增多、spill 或 occupancy 下降，独立 fast primitives 可能更快。此列表是候选方向，不是这些融合在所有框架中已有支持的声明。
3. **提交层**：复用 pipeline/资源，减少不必要的 CPU 读回与等待，观察 command buffer 粒度。不要删除必要的跨 encoder/queue 同步；异步返回更快不能算 GPU 工作更快。低层同步和内存见 [Metal 页](metal-memory-threadgroups.md)。

本机现有 [残差 RMSNorm 三路比较](mlx-residual-rms.md)：F32/BF16 的表达式、compile、fast 全部通过指定数值门，但小 M 排序反转、编译路径进程间波动明显。所读 [v0.31.2 通用融合列表](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/compile.cpp#L73) 不含 Reduce 或 RMSNorm primitive；含归约的表达式不能因包上 compile 就宣称整段单 kernel。该运行仅返回归一化结果，没有 profiler 或模型收益证据。

后续[自定义 Metal 双输出比较](mlx-custom-rms.md)同时保留归一化 y 和 residual s。M512 隔列输入时，默认连续化 custom 比原生路径更慢，显式 stride 路径较快；BF16 差异很小，小尺寸没有统一赢家。布局处理属于完整路径成本；此结果没有证明复制占比、实际 dispatch 数或整模型收益。

[RoPE 合并 Q/K 的 M4 检查](mlx-rope-qk.md)进一步表明：数学上等价的 head 拼接会改变布局和路径，360 个配置中 12 个出现分开通过、合并失败；另有两路共同的长 offset 精度失败。该轮未进入计时。concat 复制、split 后 batch stride 与下游消费成本都应计入候选，而不能从少一次 fast API 调用推导收益。

[SwiGLU 三路比较](mlx-swiglu.md)检查了一个已有 compiled helper 的激活：216 项预检查和 1512 个在线计时检查通过。M512 每组 dtype/layout 中 compiled 都有两轮快、一轮慢于普通表达式；自定义路径还改变 BF16 中间舍入。应保留已有优化作为基线，不能只取较快轮次或把不同算术路线的收益全归为融合。

## MLX custom Metal 的隐藏成本

默认 row-contiguous 保障可能生成输入拷贝；关闭后必须按真实 strides 寻址。把这种转换计入 end-to-end 路径。先对照普通 MLX 运算和 fast primitive，再用小范围自定义内核处理已知热点。[Custom Metal 文档](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)

不要把文档示例的固定 threadgroup 大小视为最优。对每线程工作量、向量宽度、tile 与边界处理分别比较，记录是否真的减少总设备内存流量及提交间隙。

## 上游线索与可证伪实验

[MLX issue #4521](https://github.com/ml-explore/mlx/issues/4521) 是关于 command-buffer 限额与 decode 开销的性能报告。它说明值得测量这个方向，但不是已合并修复或通用推荐；不要盲抄变量/阈值。先观察用户版本的 command buffer 数、CPU/GPU overlap、峰值内存和稳定性，再改变一个边界进行 A/B。

保留以下判断：如果 kernel 时间降了但请求耗时不变，热点可能不在它；如果 dispatch 数降了而带宽/occupancy 恶化，融合粒度可能过大；如果只改善预热首次运行，收益属于编译/缓存。针对原始输入的正确性、尾部、不同 shape 和持续运行都应通过，再纳入部署。
