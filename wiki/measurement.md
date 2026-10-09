# 测量：把慢分解成可检验的瓶颈

[English companion](en/measurement.md)

这是候选设计与测量方法，未经本机 GPU 实测。先固定工作负载，再比较优化；不以硬件宣传带宽估计替代真实结果。

## 固定比较对象

记录芯片具体 SKU、GPU 核数/内存、OS/SDK、框架版本或 commit、模型和 tokenizer/template、权重/KV 格式、输入输出长度、并发、采样、缓存状态、供电和热状态。不同量化模型可比较部署折中，但不能当成相同数值计算的 kernel 加速。

| 观测 | 必须区分 | 下一步候选 |
|---|---|---|
| 冷启动很慢 | 下载、加载/页换入、JIT、首轮分配 | 模型常驻、可复用 pipeline、打包 metallib；与稳态分报 |
| TTFT 高 | 排队、tokenization、prefill、首 token 采样/传输 | 分块 prefill、前缀复用、GEMM 路径 |
| 每 token 慢 | 小算子提交、权重读取、attention KV、采样回读 | 融合、量化、KV 路径；由 trace 选 |
| 并发吞吐低 | batch 未生效、slot 限制、CPU 串行 | 调度与 batching；同时看逐请求尾延迟 |
| 长上下文退化 | KV 字节、attention、分配/复制、swap | 精确 KV 预算、FA、量化与缓存管理 |

Apple 的 M5/MLX 文章给出了特定模型下 prefill 与 decode 的不同瓶颈，作为诊断起点；其设置不能代表所有 batch、MoE 与混合注意力模型。[Apple 原始研究](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)

## 三种计时不可混用

`llama-bench` 是算力路径基准，排除 tokenization/sampling；服务 TTFT 还包含请求处理和排队。MLX 是惰性执行，构图函数返回不代表计算完成；应求值需要的输出，涉及多流时等待实际生产它们的流。[llama-bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md)、[MLX eval](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.eval.html)、[同步语义](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.synchronize.html)

建议分别保存：prefill tok/s、稳态 decode tok/s、TTFT、逐 token 间隔、端到端延迟、并发总吞吐、峰值内存。明确 decode 计时是否包含首 token、EOS 和采样；服务 p95 需要足够请求，5 次 kernel 重复不能充当服务 p95。

先做预热，再交错 baseline/candidate 的重复运行；报告原始样本、median 和 [min,max]。按真实负载分别测试冷/热 prefix cache、短/长 prompt、单请求/并发、长时间稳定性。清 allocator cache 不等于清 GPU 硬件 cache；没有可证明的 flush 协议就写清缓存状态未知。采集 profiler 与无采集计时分开，避免将工具开销算为部署时延。

## 正确性与收益闭环

kernel 对外部 oracle 测尾部、非连续输入、极值、mask、不同形状和 dtype；量化另测真实任务质量。缓存看首次、复用、驱逐、满容量与位置偏移；投机解码看目标分布与接受/拒绝路径。每个候选记录失败，不用 dispatcher 把错误掩盖掉。

工程推断：单请求每 token 若要读大量不驻 cache 的权重和 KV，其吞吐会受有效带宽约束；batch 或 expert reuse 会改变字节量。用 profiler/测得的字节路径验证，不能把“模型大小÷广告带宽”当成预测器。最后的采用条件是约定质量和端到端目标一起通过。

最小结果行：`设备/版本 | 模型与精度 | prompt/output/batch/context | 冷热状态 | 改动 | oracle/质量 | 指标与计时范围 | 重复样本 | profiler 观察 | 来源或结果路径 | 未覆盖范围`。
