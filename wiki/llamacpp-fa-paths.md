# llama.cpp Metal Attention：分派、量化 KV 与临时内存

量化 KV 的存储大小、执行时是否预反量化，以及 FA 的临时分配是三个不同问题。本页按上游固定提交 [`36a7391`](https://github.com/ggml-org/llama.cpp/commit/36a73916ee0cb3b457f356066afabd47cce68884)（2026-10-07）解释源码机制；另有本机现存 binary 的 **45 项有限 CPU 对照检查**。本机 binary 的 build commit 为 `unknown`，其结果不能充当这个上游快照的运行验证。

## 先确定实际 FA 算子的输入

这里 Nq 是一个 `FLASH_ATTN_EXT` op 的 query 长度，Nk 是其 KV 长度，DK/DV 是 K/V head dim，Hq/Hkv 是 head 数，B 是张量 batch。Nq 不直接等于 server 并发数或 CLI 的 `-b`；microbatch 拆分及图构造会改变实际形状。

该快照的 [Metal 支持检查](https://github.com/ggml-org/llama.cpp/blob/36a73916ee0cb3b457f356066afabd47cce68884/ggml/src/ggml-metal/ggml-metal-device.m#L1726)要求 K/V 同 dtype，接受 F16、F32、Q4_0/Q4_1/Q5_0/Q5_1/Q8_0，BF16 另需设备能力；整个 FA 支持检查仍要求 simdgroup matrix multiply。Q 的执行路径要求 F32。CLI 接受混合 KV 参数不代表 Metal FA 会接受它们，不能把其他后端的支持范围移过来。

还要查具体 head pair 实例，不能把支持检查中的 `DK≥DV` 当成任意组合均可用。所读 [regular 实例表](https://github.com/ggml-org/llama.cpp/blob/36a73916ee0cb3b457f356066afabd47cce68884/ggml/src/ggml-metal/kernels/fa_f16.metal)列出等维 `{32,40,48,64,72,80,96,112,128,192,256,512}` 和 `96/64、192/128、320/256、576/512`；[vec 实例表](https://github.com/ggml-org/llama.cpp/blob/36a73916ee0cb3b457f356066afabd47cce68884/ggml/src/ggml-metal/kernels/fa_vec_f32.metal)对应其中 DK 为 32 倍数的 11 组。这是源码实例范围，量化 block/layout 合法性及设备支持仍须同时成立，并非全矩阵实测通过。

[构造规则](https://github.com/ggml-org/llama.cpp/blob/36a73916ee0cb3b457f356066afabd47cce68884/ggml/src/ggml.c#L5506)还要求 head 整除关系、匹配的 batch、连续 F16 mask 和合法广播；Metal 另检查 mask 的 query 轴足够长。不要直接把 bool mask、任意 stride 或一个滑窗数字替换成该 mask。

## 不是一条“FA 开启”路径

以下来自[分派与预处理代码](https://github.com/ggml-org/llama.cpp/blob/36a73916ee0cb3b457f356066afabd47cce68884/ggml/src/ggml-metal/ggml-metal-ops.cpp#L3073)，没有在当前快照下做设备计时。

| 路径 | 关键条件与要计入的工作 |
|---|---|
| 普通 vec | Nq<20 且 DK%32=0；也处理多个 query。路径包含分块部分结果与后续 reduce，不能按一次 API 调用计为一次 dispatch。 |
| regular | 普通 vec 不适用、且未选择其他路径时；可能包含 tail padding 和 mask-block 预处理。 |
| Tensor API | 需要设备能力与 shader 试编译、非 vec、受支持 head pair、足够工作量、F16 KV 或可预转换，以及 Nk%64=0。它是独立条件路径。 |
| sparse vec | 有合法 `n_kv_max` hint 和 mask、受支持的 pair/type 时，可覆盖普通 Nq 分支，使用原 dtype KV；量化类型在 kernel 内解量化，跳过预反量化。 |

Tensor 路径的 DK/DV 限于 `64/64、128/128、192/128、256/256、512/512、576/512`，工作量门为 `ceil(Nq/nqptg)*Hq*B*DK ≥ 8192`；nqptg 根据大 head 分支确定。设备的 Metal 4 标记本身不充分，[能力检查](https://github.com/ggml-org/llama.cpp/blob/36a73916ee0cb3b457f356066afabd47cce68884/ggml/src/ggml-metal/ggml-metal-device.m#L1146)还可能禁用或试编译失败。已有 M4 binary 本次日志为 `has tensor=false`，没有通过改环境强行打开它。

## 预反量化的计算与存储代价

[PR #27390](https://github.com/ggml-org/llama.cpp/pull/27390)于 2026-08-20 合入：将部分量化 K/V 展开为 F16，再运行已有 attention；V 为 K 的指定 view 时可共享展开结果。动机是减少 query block 重复解量化，同时付出转换、写临时结果及随后 F16 读取的成本。PR 的作者测量不是本机收益证据，早期提交的 type-only 条件也不能代替当前 gate。

固定快照的五种量化 KV 类型只在符合 gate 时预转换。非 sparse、DK=128 时，Nq<32 不转换，Nq≥32 转换；DK=512/576 在 Nq20–31 的非 vec 路径也转换，源码说明这还关系到 regular 的 threadgroup memory 容量。不要把 query 数 20 与转换阈值 32 混为一谈，也不要把这些源码阈值称为 M4 的实测最优点。

更容易漏掉的是分配：[`get_alloc_size`](https://github.com/ggml-org/llama.cpp/blob/36a73916ee0cb3b457f356066afabd47cce68884/ggml/src/ggml-metal/ggml-metal.cpp#L220)对 FA 输出无条件加上 pad、blk、tmp、kv_f16、idx 等需要的区域；[kv_f16 大小函数](https://github.com/ggml-org/llama.cpp/blob/36a73916ee0cb3b457f356066afabd47cce68884/ggml/src/ggml-metal/ggml-metal-ops.cpp#L3357)不按执行 gate 返回零。独立 K/V 时该项为 `align16(2*DK*Nk*Hkv*B) + align16(2*DV*Nk*Hkv*B)` 字节；识别出 V 是 K 的 view 时只保留 K 项。

例如 DK=DV=128、Hkv=8、Nk=32768、B=1、K/V 独立：

| 内容 | 按源码推算的容量 |
|---|---:|
| F16 逻辑 K+V 数据 | 128 MiB |
| Q8_0 逻辑 K+V 数据（每32值34字节） | 68 MiB |
| Q4_0 逻辑 K+V 数据（每32值18字节） | 36 MiB |
| FA 输出分配中的 F16 scratch 项 | 128 MiB |

量化块大小来自[类型定义](https://github.com/ggml-org/llama.cpp/blob/36a73916ee0cb3b457f356066afabd47cce68884/ggml/src/ggml-common.h)。同一例子 Nq=1 和32 的 F16 scratch 项相同，尽管量化输入只有后者执行预转换。这只是单 op 的一个分配分量：未含其他临时量、输出、KV capacity padding 或 allocator 生存期/复用，不是测得的全图峰值，不能直接按层数相乘或与 RSS 相加。优化容量时同时检查常驻 KV 和实际 compute buffer。

## 稀疏 mask 的 hint 必须保持语义

[`ggml_flash_attn_ext_set_n_kv_max`](https://github.com/ggml-org/llama.cpp/blob/36a73916ee0cb3b457f356066afabd47cce68884/ggml/include/ggml.h#L2506)是内部算子 hint，其契约是每行有限 mask 项数的上界，不是本页推荐的新 CLI 开关。该快照 sparse gate 还要求 mask 存在、`0<hint≤4096` 和受支持的组合。

[索引 kernel](https://github.com/ggml-org/llama.cpp/blob/36a73916ee0cb3b457f356066afabd47cce68884/ggml/src/ggml-metal/kernels/fa_aux.metal#L182)按 `isfinite(mask)` 收集位置。有限的大负数仍算有效项；若实际有限项超过 hint，代码只保留前 hint 项，没有自动退回 dense。mask 才是数值来源，不能为缩小工作量而低报上界。滑窗、sink、位置偏移和不等长 batch 都要按真实可见集合验证；本地 45 项没有调用或验证 sparse hint。

## 本地现有 binary：45 项确实执行，但范围有限

2026-10-07，Apple M4/16GB、macOS 27.0（26A428）。现存 Release/arm64 测试程序及旁边的库未重建；生成元数据与 SQL 均报 `build_commit=unknown`，没有可核实的上游源码到 binary 映射。持有协作 GPU 锁，一次调用 `test-backend-ops test -b MTL0 -o FLASH_ATTN_EXT -p <固定完整过滤器> -j 1 --output sql`，未运行 perf、模型或 profiler。

预先固定 DK=DV=64、Hkv=4、B=1、Q 为 F32；KV 同 dtype={F16,Q8_0,Q4_0}，Nq={1,3,32}。Nk=113 只测 GQA1、原布局，共9项；Nk=512 测 GQA1/4、原布局/0213 permutation，共36项。K/V 是缓存 view，mask 开启，sinks/ALiBi/softcap 关闭，prec=f32。不能把本地 Nq32 当成当前快照 F16 预转换路径已经验证。

过程退出0，**45个唯一参数串与预定集合完全一致**，所有 SQL 行均为 `MTL0/test/supported=1/passed=1`。stderr 确认 Apple M4 初始化，未见错误；独立 CPU 解析复核了完整集合和原始日志，没有执行 SQL。日志中的 pipeline 加载记录不能用来计算 dispatch 数。

所读本地测试源比较 Metal 与内置 CPU 后端，请求可选的 CPU reference 模式；FA 阈值是全张量 NMSE≤5e-4，分母为 Metal 输出能量，另有 sentinel、NaN/Inf 检查。它不是逐元素 atol/rtol 或逐行误差门；成功行不输出误差大小。按所读生成器和这些小尺寸推导，mask 为有限随机加性值，生成的 `-inf` block 数为0；没有保存 mask 数组供独立检查，不能宣称覆盖 causal/遮挡正确性。

随机 seed、输入、输出和 oracle 数组未导出，无法独立重算原数值；独立审查只验证记录完整性。未测混合 KV dtype、其他 head dim、长上下文、稀疏 hint、模型质量、性能或当前上游 commit。原始记录逻辑引用 `2026-10-07-llamacpp-fa/derived/summary.json`，来源 `local-llamacpp-fa-20261007`；原记录在仓库外且未公开，不能把此页当作可独立重放的实验包。

## 怎样用于下一轮优化

先取得实际 op shape、K/V dtype、GPU family、框架 commit 与加载日志，再分别量化常驻数据和 compute buffer。围绕源码路径变化点设计 Nq19/20、31/32、KV tail、head pair、实际 mask 和 view 的数值矩阵；这些新增边界尚未在本页本机验证。使用不同实现时须保持共同 oracle 和原容差，量化质量与算子执行错误分开检查。

性能比较应包含反量化、padding、索引、reduce、同步与下游消费，并回到相同模型的 PP/TG、KV depth、TTFT 和质量；没有稳定重复与 profiler 证据时，源码机制只用于提出候选。配合 [部署](llamacpp-deployment.md)、[Metal 调参](llamacpp-metal-tuning.md)、[防空跑验证](llamacpp-validation.md)、[量化](quantization.md)使用。
