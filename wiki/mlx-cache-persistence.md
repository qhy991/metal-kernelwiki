# 缓存保存与恢复：读回张量不等于能够续写

证据：2026-10-07 一次 M4 本地运行，MLX 0.31.2、MLX-LM 0.31.3；另核查上游固定提交 `ee19be4`。12 份缓存文件均能读回，张量与用户 metadata 一致；其中 2 份旋转标志恢复错误，恢复后的批量旋转缓存 6 次续写全部因缺字段抛异常。其余指定对照通过，不代表完整模型或跨版本恢复已验收。

## 前缀复用需要可继续执行的状态

保存 prompt cache 可以避免重复处理相同前缀；采用前先验证模型、tokenizer、模板和 token 前缀一致，并把读盘成本计入 TTFT。[KV cache 部署](mlx-kv-cache.md) 说明了复用条件。本页补充文件往返边界：K/V 数组相等、类名正确和文件加载成功，仍不足以证明控制字段、mask 与下一次更新正确。

保存时记录实际缓存类、框架版本和所处阶段；加载后分别检查公开状态、控制字段以及真实续写路径。单 token decode 与多 token prompt 追加是不同分支；各分支从原快照重新加载，用未保存的同历史缓存和独立逻辑 token 参考作对照。不要从失败对象接着尝试并把后续结果混入成功率，也不要补字段后将原始失败改报通过。

## 本机协议

Apple M4/16GB、macOS 27.0（26A428）、Python 3.14.3、NumPy 2.4.3；默认 Metal GPU，`MLX_ENABLE_TF32` 未设置。确定性合成 K/V，不加载模型或更新包。

四类缓存：`KVCache`、`RotatingKVCache`、`BatchKVCache`、`BatchRotatingKVCache`。单请求类 B=1，batch 类 B=2 且两请求等长；`H=1,D=64,float32`，旋转类 `max_size=8,keep=0`，无非零 padding。这隔离了此前 [不等长请求的 mask 错位](mlx-cache-mask-alignment.md)，没有用等长对照推翻旧失败。

先写 6 个 token；随后逐个写 3 个到 t9，再写 3-token 块到 t12。在 t6、t9、t12 各保存一份，合计 12 份。每份分别测试 N=1、N=3 续写：一个分支从相同历史重新构造，另一个分支从原文件重新加载，合计 48 次。都在同一进程完成，不是进程重启恢复或独立随机试验。

r、t、c 从 0 开始，K=`1000*(r+1)+t+1+c/128`，V=`2*r+(t+1)/16+c/1024`，实际 float32 值可精确表示。预设门槛：保存前后类、公开 state 的 dtype/shape/所有值、meta_state 和用户 metadata 相等；续写必须成功，完整 K/V 编码与 offset 正确，每个应见 token 恰好存在一次。mask 在更新前生成并求值、保存，再与更新后实际物理槽位对照；CPU 按 `0<=key<=query` 且 `query-key<window` 构造参考，逐位比较 window=8 和 3。`None` 按无屏蔽处理，bool 数组广播到 `[B,1,N,L]`。输入、原状态与 mask 在可能失败的更新前落盘，异常另存 partial state。

## 结果：状态差异与续写异常分开记录

| 缓存类 | 完整往返门通过 | 未保存对照续写 | 恢复后续写 |
|---|---:|---:|---:|
| KVCache | 3/3 | 6/6 | 6/6 |
| RotatingKVCache | 3/3 | 6/6 | 6/6 |
| BatchKVCache | 3/3 | 6/6 | 6/6 |
| BatchRotatingKVCache | 1/3 | 6/6 | 0/6，均抛异常 |

所有 12 份都成功加载，公开 state 张量、类和用户 metadata 均相同。BatchRotating 的 t6、t12 原本 `rotated=False`，读回变为 True，因此完整往返门仅 10/12 通过；t9 原本已为 True，未出现这项差异。已安装实现将标志写为字符串，再用 `bool(v[3])` 读取；非空字符串 `"False"` 会得到 True。[v0.31.3 源码](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/cache.py)

另一方面，恢复的三个 BatchRotating 对象均没有 `_lengths`，N=1 和 N=3 的更新都抛出 `AttributeError`，共 6 次。恢复构造绕过 `__init__`，而对应 state/meta_state setter 没有初始化该字段。即使 t9 的旋转标志一致，也无法完成指定续写；不能将这些异常全部归因于 bool 转换。异常前保存的 12 个窗口 mask 恰好与 live 对照一致，本轮没有观察到 bool 错误造成的 mask 差异，也没有测恢复后 attention。

异常不保证对象未变：t9 的 N3 分支在报错前已将物理顺序从 `[8,1,2,3,4,5,6,7]` 变为 `[1,2,3,4,5,6,7,8]`，`_idx` 从 1 变为 8、rotated 变为 False；未追加新 token，offset 未增长。因此失败后应保留 partial state，并从原快照另建研究分支。

42 次完成的续写均通过内容、完整可见集合、offset 和两个窗口的 mask 检查，共 84 项窗口检查；6 次异常没有返回可用于这些门的结果。整体保留 `completed_with_failed_gates`、exit 2。另观察到恢复的 BatchKVCache 缺 `_right_padding`，但本轮未调用依赖它的 finalize/right-padding 路径，因此此属性缺失不是本轮续写失败，表中通过也不覆盖该生命周期。

## 上游修复与文件格式是两个边界

[问题 #1250](https://github.com/ml-explore/mlx-lm/issues/1250) 已关闭。[PR #1778](https://github.com/ml-explore/mlx-lm/pull/1778) 于 2026-09-09 合入 `ee19be43625b9385f979de6133938f683aba6e8e`，将控制标量纳入 state 并按类型序列化，避免 bool 字符串问题。其动机还包括恢复时保留原始 padding 以支持优化的 SDPA 路径；这不是本机 Metal 收益测量，也没有证明本机安装版包含该变更。

对[该固定提交的源码](https://github.com/ml-explore/mlx-lm/blob/ee19be43625b9385f979de6133938f683aba6e8e/mlx_lm/models/cache.py)检查表明，文件 metadata 从旧的 `[cache_info, metadata, classes]` 改为 `[metadata, classes, scalars]`，读取函数未包含旧格式识别或转换分支。因此不能承诺旧缓存文件可直接复用，也不能保证不兼容时必然明确拒绝。本轮未执行新版本或跨版本文件测试，未确认首次包含变更的发行版。

同一固定提交的 BatchRotating 恢复 setter 仍未初始化 `_lengths`；这是**源码观察**，不是本轮对新版本的运行结果。仅看到 #1250 已关闭，不足以证明所有保存恢复续写路径已验收。部署候选仍需在选定版本上测试状态与后续操作，保留原文件和失败记录。

本轮未测非零 padding、keep>0、量化缓存、finalize/filter/extract、SDPA/RoPE、模型质量、服务、文件大小收益、I/O 延迟、TTFT 或跨版本兼容性；每步显式求值，不覆盖长惰性链。原始快照、数组、脚本、日志与失败存于仓库外，来源 `local-mlx-cache-roundtrip-20261007`，逻辑引用 `2026-10-07-mlx-cache-roundtrip/derived/summary.json`。原记录未随公开仓库提供，外部读者不能据本页独立复验该运行。
