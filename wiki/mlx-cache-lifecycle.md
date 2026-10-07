# 旋转 KV cache：批处理必须保留每个请求的策略

证据：MLX-LM v0.31.3 源码、上游报告与 2026-10-07 一次 M4 本地探针。已观察到直接 merge/extract 路径丢失 `keep=4`；本页没有测服务、attention 或模型质量。

## 内存优化不能悄悄改变保留规则

旋转缓存限制历史长度；`keep` 决定保留多少前缀位置，其余槽保留最近的 token。将已有缓存合并以批处理时，容量相同不代表策略兼容。需要核查缓存类型、keep、offset、有效历史和 padding；在吞吐比较之前先证明这些语义仍成立。

在所读 [v0.31.3 cache.py](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/cache.py) 中，`BatchRotatingKVCache.merge` 检查相同 `max_size`，没有承接 `keep`；`extract` 构造默认 `keep=0` 的 `RotatingKVCache`。但 [v0.31.3 generate.py](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/generate.py) 的 `_make_cache` 会拒绝模型 `make_cache()` 返回的 `keep>0` 旋转缓存。必须按入口区分：一条路径有 guard，不能推出直接 merge 也受保护，更不能说所有批处理均存在同样行为。上游 [#1631](https://github.com/ml-explore/mlx-lm/issues/1631) 也报告了这种差别。

## 本地探针和独立参考

Apple M4/16GB，macOS 27.0（26A428），MLX 0.31.2、MLX-LM 0.31.3、NumPy 2.4.3、Python 3.14.3。默认 Metal GPU；不加载模型、不修改软件包或精度开关。

每组两个请求，`H=1,D=64,float32,max_size=8`。先写 6 个 token，经 `RotatingKVCache.merge` 合并；即时 extract 只做观察，随后仍向原 batch 逐 token 追加 10 个，最后再 extract。另建独立单请求缓存作为对照。

CPU 参考在历史不超过 8 个 token 时全部保留；超过后按位置保留前 `keep` 个和最近 `8-keep` 个。对请求编号 r、token 编号 t、通道 c，K 为 `100*r+t+c/16`，V 为 `-2*(100*r+t)-c/32`。全部值可精确表示；检查完整 K/V 通道的精确数值相等、keep、offset 和 size。最终正好处于已完成的环形周期，不能把这套直接数组比较推广到任意物理环形位置。

| 输入 keep | 初次 extract | 追加后的 extract |
|---|---|---|
| `(4,4)` | 两个请求均变为 0；6-token 内容尚未丢失 | 两者均留下 `[8..15]`，应为 `[0,1,2,3,12,13,14,15]` |
| `(0,0)` | 策略与内容正确 | `[8..15]`，本组检查通过 |
| `(0,4)` | merge 接受；第二请求变为 0 | 第一请求正确，第二请求丢失应保留的前缀 |

12 个单请求对照观察均通过；所有数据有限，所有 offset/size 检查通过，终态为 `offset=16,size=8`。因此只看形状、容量、offset 或初次内容会漏掉策略损失。“混合策略必须拒绝”是本探针预先声明的要求，不是已核实的上游 API 保证；可确定的实际行为是接受后丢失了 keep=4 的策略与内容。

## 对部署候选的约束

`keep=0` 通过是另一种策略的对照，不是 keep=4 的等价修复。若工作负载要求保留前缀，不应把未经验证的合并路径用于吞吐优化。仅按 keep 分组也不能修复本例，因为 `(4,4)` 同策略合并同样失败。可提出拒绝不支持组合或改用经过验证的版本/实现等候选，但每个候选仍需独立验证跨容量、续写、mask 和模型质量；本轮没有实施这些修复或证明其速度。

此轮未执行 BatchGenerator、服务请求、attention/mask、padding、不同长度、量化缓存、extract 后的继续生成，也未测性能。原始运行与失败状态保存在仓库外：来源 `local-mlx-qvm-cache-20261007`，逻辑引用 `2026-10-07-mlx-qvm-cache/results-summary.json`。原始数组、脚本和日志未随公开仓库提供。

与 [KV cache 部署](mlx-kv-cache.md)、[服务并发](mlx-serving.md)、[边界探针进度](mlx-regression-probes.md) 配合使用。
