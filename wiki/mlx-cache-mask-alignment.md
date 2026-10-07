# 旋转缓存 mask：内容正确仍可能屏蔽有效历史

证据：2026-10-07 的一次 M4 本地运行，MLX 0.31.2、MLX-LM 0.31.3。90 项窗口检查中，4 项原生 mask 与 attention 输出失败；返回的缓存内容、应见 token 完整性和 offset 全部通过。适用范围是下面的直接 API 轨迹，不是整模型、服务或当前上游版本的普遍结论。

## 优化前先验证 mask 与返回槽位对齐

旋转缓存可以减少历史存储，但返回 K/V 的物理顺序会随逐 token 更新和多 token 拼接改变。调用时先创建本次 mask，再 `update_and_fetch`，最后将 mask 与本次返回的 K/V 对照。窗口包括当前 token；多 token 更新可暂时返回 `max_size+N−1` 列，不能把每次返回宽度强制限制为 `max_size`。[缓存实现](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/cache.py)、[窗口语义](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/base.py)

CPU 参考维护独立的逻辑 token 历史，从 K 的可逆编码识别返回列；不复制实现中的 roll、索引或 padding 公式。有效 query 为 q，返回 key 为 k 时，允许访问的条件为 `0 <= k <= q` 且 `q-k < window`。同时要求完整应见集合 `{max(0,q-window+1),…,q}` 中每个 token 恰好存在一次，避免缓存漏掉历史后，mask 仍对剩余内容“自洽”。

## 本地配置与预设门槛

Apple M4/16GB、macOS 27.0（26A428）、Python 3.14.3、NumPy 2.4.3；默认 Metal GPU，`MLX_ENABLE_TF32` 未设置。无模型、软件包更新或性能测量。

- `B=2,H=1,D=64,float32,max_size=8,keep=0`。这里单独检验 keep=0，不改变先前 [keep=4 策略损失](mlx-cache-lifecycle.md) 的失败结论。
- 初始有效长度为 `[3,6]`、`[6,3]`、`[6,6]`；每组使用两个入口：直接构造左填充 batch，以及分别构造单请求缓存后 merge。
- 每条轨迹依次追加 `[1,1,1,3,1,2,1]` 个 token；每步检查 `window=8` 和 `3`。直接入口另检查初始块，因此共 `6×7×2+3×2=90` 项。它们来自一个进程，不是 90 次独立随机试验。
- r、t、c 分别是从 0 开始的请求、逻辑 token 和通道编号。K=`1000*(r+1)+t+1+c/128`，V=`2*r+(t+1)/16+c/1024`，padding K/V 为零；检查完整通道编码、offset、应见 token 集合和 mask 的 bool 类型、形状、逐位相等。
- Q 全零、scale=1；独立 CPU float64 参考为应见 V 的均值。有效 query 要求有限且 `max_abs<=1e-5`，这是此合成探针的预设门槛。左填充的无效 query 只检查全 False mask，不对其 SDPA 输出规定数值。
- 另用同一返回 K/V 和 CPU 参考 mask 调用 SDPA，作为区分 mask 与数值计算的对照。对照通过不能替代原生路径通过。

## 观察到的失败

| 检查 | 通过 / 总数 |
|---|---:|
| 原生 mask 逐位一致 | 86 / 90 |
| 原生 mask 的 attention 输出 | 86 / 90 |
| 完整 K/V 编码、应见集合、offset（各项） | 90 / 90 |
| CPU 参考 mask 的 attention 输出 | 90 / 90 |

4 项失败分别来自两个入口的 `[3,6]` 和 `[6,3]`，都发生在三次单 token 追加之后的 `N=3,window=8`。短请求此时的 query 为 `[6,7,8]`，返回 key 槽位为 `[PAD,0,1,2,3,4,5,6,7,8]`。q=6 应见 0…6，q=7 应见 0…7，q=8 应见 1…8。原生 mask 把前两行的 token 0 错误遮住，共 **8 个错误 mask 位、8 行错误 query 输出**，最大绝对误差均为 **0.03125**。

该合成输入中，未失败原生项和所有参考 mask 对照的最大误差均为 0。`window=3` 的 45 项、等长请求的 30 项均通过；两组存在重叠，不相加作为新样本。小窗口本来就看不到 token 0，因此它通过不能证明 window=8 正确。交换请求位置仍复现，表明问题随短请求而非固定 batch 行出现。

记录与已安装源码的解释一致：失败步更新前 `_idx=1,rotated=True`，短请求 `left_padding=2`；`make_mask(3)` 据旧索引判断无需裁剪。随后多 token 更新先将缓存恢复时间顺序，再裁去一列，使实际 padding 降到 1；mask 仍屏蔽物理列 1，而该列已经是 token 0。这是本次观测支持的源码诊断，未修改实现或验证修复版本。

## 对优化候选的影响

对本版本、上述不等长旋转后多 token 续写路径，不能仅凭存储/offset 正确就验收 batching 或 chunking。保留真实窗口和缓存策略，覆盖单 token 与多 token 更新的切换，并让 mask 对照实际返回列；只检查 mask 的固定常量不足以证明它与随后返回的 K/V 对齐。所读 [v0.31.3 测试](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/tests/test_prompt_cache.py) 有相关切换场景，但最后的常量断言未继续更新并检查 token 可见性。

本轮每步显式求值元数据及输出，不覆盖长期惰性依赖积累。未测 keep>0、右填充、filter/extract 后续写、保存恢复、RoPE、量化缓存、非零 Q、服务入口、模型质量或性能；等长对照通过也不是所有等长配置的保证。CPU 参考 mask 是诊断工具，尚未作为部署修复评估其开销或普适性。

原运行以 `completed_with_failed_gates`、exit 2 结束；90 项数组、执行脚本、安装源码快照和日志在仓库外保留。来源 `local-mlx-cache-mask-20261007`，逻辑引用 `2026-10-07-mlx-cache-mask/derived/summary.json`。原始记录未随公开仓库提供，外部读者不能据本页独立复验该运行。

相关：[KV cache 部署](mlx-kv-cache.md)、[缓存策略生命周期](mlx-cache-lifecycle.md)、[探针进度](mlx-regression-probes.md)。
