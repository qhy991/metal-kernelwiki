# llama.cpp：Metal FA 调参与后端证据

[English companion](en/llamacpp-metal-tuning.md)

核对日期：2026-10-07。适用于 profiler 已把热点定位到 Metal attention/kernel 的后端开发；普通部署先读 [部署页](llamacpp-deployment.md)。调参会占用 GPU，以下命令是工作方案，本次知识整理没有执行它们。

## 1. 可复用经验：优化按 GPU 家族与形状落地

PR #26570 的 FA-vec `(Q, NE)` 调优在 2026-08-24 合入，包含离线 tuner、热漂移处理和基线保护；它是具体方法来源，不是所有 Mac 的收益承诺。[合入记录与变更](https://github.com/ggml-org/llama.cpp/pull/26570/files) 历史 PR 中的 `efeda76b948f59ee52ea20db640bc4cf3dfe8ac1` 可定位当时的补充调优提交；它不是本次浏览的 master 版本。

初期讨论按设备收集结果，随后比较 GPU family 分组；当前文档已要求表按 Apple GPU family 归档。不能把旧讨论的设备行原样贴入新版本，也不能只用 M1/M4 结果推断 M5。[上游讨论](https://github.com/ggml-org/llama.cpp/discussions/27668)

## 2. 小范围 sweep，再检查数值与应用

```sh
# 在已固定 commit 的独立 llama.cpp checkout；先检查实际参数
cmake -B build-metal -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
cmake --build build-metal --target ggml-metal-tuning test-backend-ops -j
./build-metal/bin/ggml-metal-tuning --help
# 确认当前帮助支持后，缩小到需要的 dtype/head dim
./build-metal/bin/ggml-metal-tuning fa-vec --dtype f16,q8_0 --dk 128 \
  > /absolute/path/external-results/fa-rows.txt \
  2> /absolute/path/external-results/fa-sweep.log
./build-metal/bin/test-backend-ops test -o FLASH_ATTN_EXT -b MTL0
```

当前 tuner 依 dtype、head size、KV depth、batch width 比较候选，保留逐点不比基线慢的配置，并用 anchor 检查热漂移。完整 sweep 需数小时；保留 timing 日志和被拒绝配置，不能只保留赢家行。[调优文档](https://github.com/ggml-org/llama.cpp/blob/master/tools/tuning/README.md)

关键覆盖限制：tuner 不做数值验证；退出 0 不表示性能或正确性通过。文档所列 `test-backend-ops` 强制配置覆盖 `dk=128/576`，其余头维度没有相同自动数值覆盖。对目标模型实际形状补齐测试，然后做外部任务质量、长上下文和端到端回归。微内核胜出只构成候选证据。[调优边界](https://github.com/ggml-org/llama.cpp/blob/master/tools/tuning/README.md)

## 3. Metal Tensor API 是条件路径

当前设备代码先检测 Metal 4 能力并试编译 tensor kernel，再选择路径；较旧芯片默认关闭 tensor API。不要把 `GGML_METAL_TENSOR_ENABLE` 当成通用加速开关。打包时还要核对 `ggml-tensor.metallib`，缺失可能关闭该路径。记录 `has tensor`、GPU family、编译器/OS 和实际 dispatch，再比较 PP、TG、量化与 batch；不要只凭 API 版本宣称使用了更快硬件。[设备和库加载代码](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-device.m)

调参前先读固定快照的 [FA 分派与内存](llamacpp-fa-paths.md)：普通 vec 的 query 阈值、量化 KV 预转换阈值及 sparse 覆盖不是同一条件。分配中的 F16 scratch 不等于每次都执行转换；更不能任意调小 `n_kv_max`，它约束 mask 的实际有限项数量。

## 4. 报告热点、干预与失败面

用 Metal GPU trace 定位 FA、matvec/matmul、图提交或同步开销，再限定一个干预。报告 shape、dtype、KV 长度、slot/batch、warmup、热状态、计时范围和分布，并给出 shader 数值容差与真实模型指标。需要 CPU/Metal 算子对照、perplexity 和 bench 回归；新增算子测试也应随实现更新。[上游贡献检查](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md)

如果头维度、精度或 GPU 家族超出检查范围，明确记录“未验证”；保留较慢和错误候选，不能用启发式 dispatch 隐藏失败。该页不授权修改宿主项目的冻结 Compiler/Executor，也不替代其 Corpus Gate 与人工发布审批。
