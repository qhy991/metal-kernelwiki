# MLX Serving：先检查 batch 路径，再搜索并发

证据状态：上游源码描述；核查于 2026-10-07。本文针对 MLX-LM 所见 `main`，不承诺其他服务器具有相同特性。本页没有本机服务或吞吐测量。

## 建立请求级目标

固定模型、请求到达率、输入/输出长度分布、采样、缓存命中率与并发上限。保留单请求基线，再逐步增加并发。报告总输出 tok/s、完成请求/秒、队列时间、P50/P95 TTFT、TPOT、错误率与内存。高聚合吞吐不保证某条请求更快；先满足服务延迟与质量目标，再比较吞吐。

## 确认进入批处理

当前 server 将 `decode_concurrency` 传给 `completion_batch_size`，`prompt_concurrency` 传给 `prefill_batch_size`，并向活动 batch 插入请求。`_is_batchable` 要求模型支持 batch、请求 seed 为 None、CLI `kv_bits` 为 None；不满足会走顺序路径。新请求若与当前 batch 不兼容，会先排空该 batch。[server.py](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/server.py)

因此先观察实际路径，再谈参数效果。不能声称 KV 量化、固定 seed、所有模型、所有 speculative 配置都能同时使用连续批处理；也不能为了进入 batch 偷改业务要求的采样语义。安装版本不同则重新检查条件。

## 并发与缓存一起预算

分别扫描 prompt 并发、decode 并发和 prefill 步长，记录各请求的延迟分布，尤其加入一个长 prompt 对其余请求的影响。检查活跃 KV 与可复用 prompt cache 的总预算；所见实现会从 prompt cache 预算减去活动 batch 的 cache 字节。[缓存预算实现](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/server.py)

分别测试缓存命中与未命中，避免热前缀掩盖冷请求成本。对每个候选执行混合长度、取消请求、完成后新请求加入及超长输入测试；记录未覆盖的行为。后两类是本页提出的验证方案，不是上游已验证的性能结论。

## Speculative decoding 是独立候选

长输出可尝试兼容的小 draft model 与不同 draft token 数。当前 `speculative_generate_step` 要求目标 prompt cache 可 trim，并在拒绝后回退缓存，还能标识 token 是否来自 draft。[生成实现](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py)

先确认 tokenizer、词表和采样语义兼容，再测接受比例、draft 与目标验证成本、额外权重/KV 内存、整段输出时间与质量。低接受率或内存压力可使结果变慢；没有这些数据不能宣称提速，也不能直接把单请求收益外推到批处理。

以下为未执行的选项发现入口：

```sh
python -m mlx_lm.server --help
```

部署基线见 [Deployment](mlx-deployment.md)，缓存边界见 [KV cache](mlx-kv-cache.md)。
