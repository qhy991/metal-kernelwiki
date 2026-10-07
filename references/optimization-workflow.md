# Metal KernelWiki 优化流程

用来源可追溯的知识库，把 Metal 上的 LLM 性能问题落实为可验证的优化候选。
资料检查日期：**2026-10-07**。包含文档、上游实现研究与有限的 **M4/MLX 本地原语验证**；尚无整模型部署收益结论。
最新 API、CLI、默认值与组合支持必须以用户安装的版本为准。

## 使用流程

1. 从已有上下文取得芯片型号/GPU family、统一内存、macOS/Xcode、框架版本、模型/量化格式、prompt/output 长度、并发，以及用户关注的 TTFT、decode 或吞吐。缺少哪项会改变下一步才询问；否则先读代码、日志和配置。
2. 先读 [测量与诊断](../wiki/measurement.md)。分开冷启动、prefill、decode 和服务排队；不能用一个 tok/s 指标决定所有优化。
3. 用下面的检索工具选最相关的 1–3 页，再追溯其来源。保留用户的框架选择；仅在证据支持时建议比较其他运行时。
4. 提出少量机制不同的候选，写出适用条件、预期改变的瓶颈、正确性风险和验证方法。优先使用框架已支持的快速路径；热点证据指向 kernel 时再写 MSL。
5. 在任务已授权、项目 gate 已通过的范围内实施和测量。先外部 oracle，再相同条件的端到端 benchmark；profiler 用来解释变化，不以采集时延代替正常运行时延。
6. 输出“诊断证据 → 改动 → 结果/未验证项 → 下一步”，附页 ID、本地路径与关键一手链接。文档支持某 API 不等于该优化在目标设备上更快。

## 检索

以下路径相对于知识库根目录，脚本自动定位知识库；可用绝对脚本路径从任意目录执行。
只需 Python 3.9+ 标准库，不需要安装 MLX、加载模型或联网。

```bash
python3 scripts/query.py "MLX 长上下文 decode 慢"
python3 scripts/query.py --engine llama.cpp --tag flash-attention
python3 scripts/query.py --symptom launch-overhead --json
python3 scripts/get_page.py mlx-kv-cache --follow-sources
python3 scripts/get_page.py metal-tensors
python3 scripts/validate.py
```

支持 `--engine`、`--type`、`--tag`、`--symptom`、`--limit`、`--json`、`--paths-only`。
检索是中英关键词和别名匹配，不是语义搜索。未命中时拆关键词或用标签，不编造不存在的页。

| 任务 | 先读 |
|---|---|
| 整体部署、首 token 慢 | [测量](../wiki/measurement.md)、[MLX](../wiki/mlx-deployment.md)、[llama.cpp](../wiki/llamacpp-deployment.md) |
| 长上下文、内存压力 | [KV cache](../wiki/mlx-kv-cache.md)、[量化](../wiki/quantization.md)、[Metal 内存](../wiki/metal-memory-threadgroups.md) |
| 服务并发、prefix reuse、投机解码 | [MLX serving](../wiki/mlx-serving.md)、[llama.cpp serving](../wiki/llamacpp-serving.md) |
| 小算子、CPU/GPU 间隙 | [MLX 执行](../wiki/mlx-execution.md)、[融合](../wiki/fusion.md)、[Profiler](../wiki/metal-profiling.md) |
| Attention/GEMM/MoE 热点 | [Attention](../wiki/attention.md)、[GEMM/MoE](../wiki/gemm-moe.md)、[Metal 调优案例](../wiki/llamacpp-metal-tuning.md) |
| M5、Metal 4、TensorOps | [能力检查](../wiki/metal-capabilities.md)、[Metal tensors](../wiki/metal-tensors.md) |
| 工具链缺失、trace 导出与计时范围 | [运行时测量](../wiki/runtime-measurement.md)、[M4 本地记录](../wiki/local-mlx-m4.md) |
| 量化 matmul 的 batch 精度、prefill 路径 | [数值与性能验证](../wiki/quantized-matmul-validation.md)、[M4 本地记录](../wiki/local-mlx-m4.md) |
| async、等待与数据依赖 | [求值边界](../wiki/mlx-async-evaluation.md) |
| float32、singleton GQA 精度 | [数值诊断](../wiki/mlx-float32-precision.md) |
| packed、逐次反量化与常驻 dense 的取舍 | [三路径时间/内存比较](../wiki/mlx-qmm-path-comparison.md) |
| 多行 qvm 行错误、batch 前缀一致却算错 | [量化数值验证](../wiki/quantized-matmul-validation.md) |
| rotating keep、合并后丢失前缀 | [缓存生命周期](../wiki/mlx-cache-lifecycle.md) |
| 左填充、旋转后 chunk 续写、mask 错位 | [mask 与返回槽位对齐](../wiki/mlx-cache-mask-alignment.md) |
| 保存缓存、加载后报错、升级后的文件兼容 | [保存恢复与续写](../wiki/mlx-cache-persistence.md) |
| 残差 RMSNorm、compile 与 fast 的精度及延迟 | [三路数值/计时比较](../wiki/mlx-residual-rms.md) |
| 自定义 Metal、非连续输入复制、双输出 RMSNorm | [布局与融合的成本](../wiki/mlx-custom-rms.md) |
| 边界探针已测与未测范围 | [探针进度](../wiki/mlx-regression-probes.md) |
| llama.cpp 测试空跑、数值或计时范围不明 | [验证流程](../wiki/llamacpp-validation.md) |

## 必须保留的判断边界

- Metal GPU、GPU 内的 Neural Accelerators、独立 ANE 是不同概念。不要把 Metal 4 API 可用性等同于所有芯片都有相同矩阵吞吐。
- Prefill 常有较高计算密度；单请求 decode 常受权重/KV 访存影响。它们是待 profiler 检验的假设，不能覆盖所有模型、MoE、batch 和 context。
- 权重量化、KV 量化、滑动窗口、prefix cache、投机解码、continuous batching 是不同机制。逐项确认组合支持；截断上下文会改变语义，量化需质量评估。
- 统一内存可避免 CPU/GPU 之间的部分显式数据搬运，但不消除格式转换拷贝、执行依赖、临时分配、页换入或内存竞争。不要自动提高系统 wired-memory 限额、关同步或开启未核实的实验变量。
- 若项目限制 clean-start/direct authoring 的参考访问，先遵守其 contract。`upstream-code` 和 PR diff 仅在允许阅读现有实现时打开；普通官方 API 文档与黑盒测量可单独使用。本技能不授权下载模型、发布服务或发起正式 GPU campaign。
- 项目存在 frozen Compiler/Executor、外部审批或 Evidence custody gate 时照常遵守；研究技能与它们的 release 无关。不得把 CUDA/CUPTI/L2 flush 协议宣称为 Metal 测量协议。

## 证据与维护

[来源与页面目录](../data/catalog.json) 保存页面、来源、检查日期、版本说明和别名；正文按主题存于 `wiki/`。
`documented` 表示官方机制说明，`source-reported` 表示上游实现/作者报告，
`inferred` 表示推断，`experimental` 表示待验证方案，`locally-measured` 表示有原始记录的限定设备/版本观测。一个原语通过不等于整模型正确；导出 trace 不等于已经解析 profiler。正文中的候选仍需在目标工作负载上验证。
性能数字需要完整设备、OS/框架版本、模型/精度、shape/batch/context、指标、重复与计时范围、来源；缺字段就不作可比速度结论。

扩充或刷新时读 [来源与维护规则](../MAINTENANCE.md)；典型检索见 [使用示例](examples.md)。
