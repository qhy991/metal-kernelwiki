# llama.cpp：并发、前缀缓存与推测解码

核对日期：2026-10-07。适用于已正确运行的 Metal 服务；这些是待验证候选，收益取决于请求分布。首先固定 prompt/template、输入输出 token 长度、并发、缓存冷热、采样和质量门槛。命令只示意，执行前核对该版本 `--help`，模型和输出路径放在 checkout 外。

## 1. 并发配额与 batch 不是同一件事

`-np` 控制服务 slot，`-b` 是逻辑 batch 上限，`-ub` 是物理 microbatch；连续 batching 允许合并活跃请求。固定并发值再扫 batch，避免自动 slot 策略改变实验。[server 参数](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)

```sh
# 将绝对路径替换为已具备的模型；这不是经过硬件验证的最佳参数
./build-metal/bin/llama-server -m /absolute/path/model.gguf \
  --host 127.0.0.1 --port 8080 -ngl all -fa on \
  -c 8192 -np 1 -b 1024 -ub 512 --cont-batching --metrics
```

分别测 1、2、4 个 slot 的端到端延迟和总输出吞吐。读取启动日志里的实际每请求上下文与 KV 配置，不要对所有版本套用 `c/np` 公式。当前 unified KV 与 `--kv-unified-per-slot` 会改变容量关系。[server 上下文选项](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)

低层扩展测试可用 `llama-batched-bench` 的 `-npl` 扫并行序列；`-pps` 的共享 prompt 与独立 prompt 是两种不同工作负载。独立模式需要约 `B*(PP+TG)` KV token，共享模式约 `PP+B*TG`；都不能直接等同线上排队。[batched bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/batched-bench/README.md)

## 2. 缓存收益必须分冷、热两组

`cache_prompt` 可复用 token 前缀；保持稳定 system prompt、模板和历史前缀，记录实际复用 token。server 的 host prompt cache 用 `--cache-ram` 限额，PR #16391 于 2025-10-09 合入，通过 RAM 中保存的状态减少重新 prefill。[缓存实现来源](https://github.com/ggml-org/llama.cpp/pull/16391) 它消耗同一台 Mac 的内存；对照 `--cache-ram 0` 与有界容量，避免不相关请求驱逐活跃对话。重复请求加速只能报告为热缓存收益；首请求、不同前缀、缓存满和混合请求都要验证。

## 3. 推测解码按任务分布选择

重复文本/代码可从 `--spec-type ngram-simple` 起步；它不需要第二个模型。独立草稿模型用 `draft-simple`，小步扫描草稿长度；MTP/EAGLE/DFlash 等有模型特定条件，不能任意互换。[推测文档](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md) 当前实现检查 target/draft 词表兼容；草稿增加的权重、KV 与验证开销必须计入。[兼容检查](https://github.com/ggml-org/llama.cpp/blob/master/common/speculative.cpp)

```sh
# 以下参数附加到同一 server 基线上，作为两个独立实验
# A: --spec-type ngram-simple
# B: --spec-type draft-simple -md /absolute/path/draft.gguf --spec-draft-n-max 3
python tools/server/bench/speed-bench/speed_bench.py \
  --url localhost:8080 --bench qualitative --category coding,math \
  --osl 256 --concurrency 1 --limit 8 \
  --output /absolute/path/external-results/baseline.json
```

先无推测保存基线，再开启一个候选并保存另一个 JSON；保持数据、长度、采样和并发相同。SPEED-Bench 可记录逐请求结果、分类型延迟、prefill/decode 与 draft 接受率；吞吐实验的客户端并发应与服务 slot 配合。[SPEED-Bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/bench/speed-bench/README.md)

接受率只是诊断：低接受率、额外 draft 延迟或内存压力都可能抵消收益。逐类验质量与延迟；代码加速不能外推高熵文本。禁止使用 synthetic acceptance 作为有效输出或部署加速证据，上游明确它可接受不匹配 target 的 token。[推测模式边界](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)
