# llama.cpp：Metal 部署、容量与阶段基准

核对日期：2026-10-07。适用于 Apple Silicon 上的 GGUF 推理；网页 `master` 会变化，命令是待执行配方，不是本机实测结论。先固定 llama.cpp commit、GGUF 名称及量化、macOS、芯片和内存；用对应二进制的 `--help` 核对参数，不能拿 server 帮助代替 bench 帮助。

## 1. 先确认真正运行 Metal

macOS 构建默认启用 Metal；显式写出开关便于复现。CPU 基线用 `-ngl 0`。不要把编译成功等同于 GPU 已承担预期算子。[构建文档](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)

```sh
# 在独立 llama.cpp checkout 中；不改宿主项目的冻结源
git rev-parse HEAD
cmake -B build-metal -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
cmake --build build-metal --config Release -j
./build-metal/bin/llama-server --version
./build-metal/bin/llama-server --help
./build-metal/bin/llama-server --list-devices
./build-metal/bin/llama-bench --help
```

从加载日志核对设备、实际 offload 层数、计算/KV 缓冲、FA 选择；异常 CPU 回落要定位到算子。当前 server 接受 `-ngl all`、`auto` 或数字，旧版本及 bench 可能只接受数字。`99` 是请求上限，不保证每个模型全层卸载。[参数入口](https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp)

## 2. 把 prefill、decode 和上下文深度分开

`llama-bench` 的 `pp`、`tg`、`pg` 测不同阶段；`-d` 预填 KV 深度；JSON 包含重复样本。它不计 tokenizer 和 sampler，所以不能充当用户端 TTFT。下例先跑单一配置；再逐项比较 `-fa off/on`、`-ub 256/512/1024`，同时保持 `-b >= -ub`，避免笛卡尔积失控。[bench 语义](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md)

```sh
# MODEL 指向现有 GGUF；OUT 指向 checkout 外的实验目录
MODEL=/absolute/path/model.gguf
OUT=/absolute/path/external-results
./build-metal/bin/llama-bench -m "$MODEL" -ngl 99 -fa on \
  -p 512 -n 128 -d 0,4096 -b 1024 -ub 512 -r 5 -o json \
  > "$OUT/bench.json" 2> "$OUT/bench.log"
```

失败模式：只报 `tg128` 隐藏长上下文退化；把重复均值当 p95；热机前后混比；吞吐增加却让单请求尾延迟恶化。保留逐次结果，交错比较候选与基线，并在真实请求上记录 TTFT、逐 token 延迟及总延迟。每次只改变一个主要因素，注明样本数与波动；若性能差异小于重复噪声，结论应为未分胜负。

## 3. FA 与 KV 量化是带约束的候选

从 F16 KV 起步，确认 FA 激活后再测试 `-ctk q8_0 -ctv q8_0`，最后才探索更激进格式。量化 V 在当前初始化代码中要求 FA；参数被接受不代表所有模型头维度/算子均有同样支持。[初始化检查](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-context.cpp) 对每一组合做长短上下文外部任务验证、perplexity/输出质量检查及内存测量；更少字节不保证更低延迟。

固定上游快照的 [FA 路径分析](llamacpp-fa-paths.md)进一步区分 vec、regular、Tensor 与 sparse：量化 KV 的 F16 预转换有 query 长度等 gate，但单 op 分配需求仍预留 F16 scratch。存储压缩比不能直接当成 compute buffer 缩减或速度收益；支持检查还要求 K/V 同 dtype 和合法 head pair。

## 4. 统一内存需要总预算

Metal 后端记录 `has_unified_memory`、`recommendedMaxWorkingSetSize` 和分配量；推荐工作集不是硬容量，超过它可能仍分配成功。[设备实现](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-device.m) 为权重、KV、计算临时量、host prompt cache、draft 和系统留余量；记录 RSS、Metal 分配、内存压力和 swap，不能把前两者直接相加为物理占用。

当前模型加载入口为 `--load-mode`；`auto/mmap/none/mlock/mmap+mlock` 含义不同，按本地帮助做冷启动与稳态对照。[加载参数](https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp) 优先减少不必要的上下文、并发和缓存，再调整量化；不要把提高系统 wired-memory 限制写成通用优化。
