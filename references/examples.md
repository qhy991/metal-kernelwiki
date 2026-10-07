# 查询示例

在知识库根目录执行；已安装 skill 可使用 `python3 scripts/wiki.py query ...` 或 `get ...`。这里只运行离线检索，不会加载模型或启动服务。

| 问题 | 检索命令 | 应得到的行动 |
|---|---|---|
| MLX 长上下文越来越慢 | `python3 scripts/query.py "MLX 长上下文 decode"` | 看 KV、attention、内存；分清容量/访存/执行问题 |
| 首 token 太慢 | `python3 scripts/query.py "首 token TTFT prefill"` | 拆冷启动/排队/prefill，再找 cache 或 GEMM |
| llama.cpp 要开 FA 吗 | `python3 scripts/query.py --engine llama.cpp --tag flash-attention` | 查看版本与模型条件；确认实际路径，再测 on/off |
| M4 能用 M5 TensorOps 配方吗 | `python3 scripts/get_page.py metal-tensors --follow-sources` | 区分 API、OS、family 与硬件，不承诺相同收益 |
| GPU 经常空着 | `python3 scripts/query.py --symptom launch-overhead` | trace CPU/提交/同步，再考虑融合和批处理 |
| MoE gather 慢或尾部错 | `python3 scripts/get_page.py gemm-moe --follow-sources` | 检查索引契约、形状边界、版本和上游状态 |
| 我只允许 clean-start | `python3 scripts/get_page.py measurement` | 先遵守参考访问 contract；不自动下载实现/PR diff |

对检索不到的问题可用 `rg` 搜正文；仍无证据时明确本库缺项，再联网查一手来源。不要把本库单机覆盖扩展为跨机器集群能力，或把 NVIDIA profiler 命令改个名称用于 Metal。
