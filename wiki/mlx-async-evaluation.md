# MLX 求值边界：依赖链延迟与独立任务吞吐

证据：官方 API、2026-10-07 的 M4/MLX 0.31.2 有界本地观察。`async_eval` 是待按任务验证的执行策略，不能从函数返回耗时直接判断推理速度。

## 改变的是何时提交、何时等待

MLX 惰性构图；`eval` 提交并等待结果，`async_eval` 调度求值，调用方仍需等待最终结果。在 [v0.31.2 求值实现](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/transforms.cpp) 中，异步路径也可能因活跃任务数或内存压力等待，不能命名为“纯 CPU 构图”；这是对应版本源码机制，未追踪本机每次等待的原因。`synchronize` 只等待指定 stream 已提交的工作，不会替代未提交图的求值。[eval](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.eval.html)、[async_eval](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.async_eval.html)、[synchronize](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.synchronize.html)

比较前先画清数据依赖：下一步需要前一步 GPU 数组，与下一步必须由 CPU 读取该值后决定，是不同的限制。保存异步返回前区间和最终完成区间，每次新建图；重复求值已经物化的同一结果不能衡量重复计算。

## 四种路径有两种工作负载

| 路径 | 实际工作 | 结束点 |
|---|---|---|
| dependent_sync | `x=tanh(x@W)` 连续 32 步，每步 eval | 最后结果已完成 |
| dependent_async | 同一依赖链，每步 async_eval | 最后 eval 完成 |
| dependent_tail | 先建立完整的同一依赖链 | 一次尾部 eval 完成 |
| independent_tail | 32 个不同 X 各自只算一步 | 全部结果 eval 完成 |

前三项可比较同一数学工作负载的完成延迟；第四项刻画独立任务处理，不能除以 32 就称为逐 token decode。完整建图可行的前提是循环不需要中间主机决策；真实生成包含采样、停止条件、KV 更新和控制流，必须另测。

## 本地观察与数值门

环境：Apple M4/16GB、macOS 27.0（26A428）、MLX 0.31.2、NumPy 2.4.3、Python 3.14.3，默认 GPU stream。CPU PCG64 seed=13，先生成 `X:[1,256]=0.1*normal`，再生成 `W:[256,256]=I+normal/64`，均存为 float32。独立输入 `X_i=X+i/1024` 在计时前已准备并求值。

参考由实际存储输入转为 CPU float64 计算，依赖链执行 32 步；独立任务各执行一步。预先设定 `atol=1e-5,rtol=1e-4`；该门槛是此探针的准确性目标，不是 MLX 的通用承诺，也没有模拟逐步 float32 舍入。四个模式在三个独立进程均通过，依赖链最大绝对误差约 `1.251e-6`。每个计时结果在计时外再次检查，全部通过。

每进程每模式预热 3 次，按轮换次序采集 12 次。测量主机建图、提交和完成等待，输入/权重准备、CPU 验证与日志在计时外。下表保留全部三个进程；中位数的范围与原始样本范围是两种统计量。

| 路径 | 三个进程中位数的 median [min,max]，ms | 全部 36 个样本的 [min,max]，ms |
|---|---|---|
| dependent_sync | 11.618 [11.491,11.839] | [4.683,16.077] |
| dependent_async | 4.160 [4.031,4.285] | [1.165,7.612] |
| dependent_tail | 1.748 [1.714,2.556] | [0.318,7.110] |
| independent_tail | 0.629 [0.609,0.695] | [0.248,1.345] |

这组结果支持“求值边界值得检查”，没有证明整模型加速或某个 kernel 更快。三个进程的依赖链中位数均为 sync > async > tail，但单个样本波动明显。没有 GPU timestamp 或 profiler，不能把差异拆解为同步、命令缓冲区数量或融合的贡献。

**环境限制**：第一进程与助手的 CPU 安装器测试重叠；后两进程也未隔离其他应用或控制功耗、温度。热输入/allocator，无硬件 cache flush；协作式锁只避免合作任务同时占用 GPU。样本之间包含 CPU 数值检查和 JSON 输出。这不是隔离 benchmark。三个进程重复同一 seed，没有增加输入分布覆盖。

本地来源 ID：`local-mlx-boundaries-20261007`。逻辑原记录引用为 `2026-10-07-mlx-boundaries/summary.json`，汇总三轮结果并指向 `run-a/b/c` 的原始结果；脚本保存在对应 run 目录的 `probe.py`；文件在仓库外，未公开，外部读者不能据本页重放原运行。

## 应如何用于优化

若 trace 显示主机频繁读取中间 GPU 值，先确认哪些等待是语义要求，哪些只是日志、`.item()` 或循环结构引入的；提出保留依赖、减少不必要读取的候选。每个候选固定输入和工作量，先过外部 oracle，再对目标模型的 TTFT、decode、内存与采样质量比较。异步方案必须计算尾部等待，不能用提交耗时替代完成延迟。

与 [MLX 执行](mlx-execution.md)、[测量合同](measurement.md) 配合使用。多流、真实服务并发和主机决定下一 token 的生成循环尚不在本页实测域内。
