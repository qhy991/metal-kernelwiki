# GEMM、GEMV 与 MoE：按真实 shape 选择算法

[English companion](en/gemm-moe.md)

适用于 matmul/quantized matmul/expert gather 热点。上游 PR 是方法来源；这里没有复现其速度，也不把 PR 描述中的旧开关当成现行 API。

MSL 层的 [矩阵乘与累加精度](msl-matrix.md)提供公开 API 的有界示例、Steel tile/fragment 机制及 M4 合成检查；它没有验证本页的模型吞吐或 MoE 路由。

[MSL tile 共享与直接装载](msl-tiles.md)另比较四个原创矩阵kernel和原生F32 matmul，记录尾块、非连续输入及主机完成时间波动；M=1的direct路线实际回退staging，不能据名称判断decode收益。

[MSL GEMV](msl-gemv.md)进一步比较沿输出、沿K、子组与两级归约，核对MLX/llama.cpp的布局与精度条件；保留长K抵消失败及相邻K配对候选的独立验证。浮点累加类型相同仍可能因顺序改变质量，本轮未测decode速度。

## 用工作分解建立候选

| 形状/阶段 | 候选 | 要测的额外成本 |
|---|---|---|
| 单 token、小 M 的 dense decode | GEMV、量化权重融合解码、连续读取 | 权重字节、寄存器与 reduction |
| 多 token prefill/连续 batch | tile GEMM、已支持的 TensorOps | padding、转换、shape 门槛、临时内存 |
| 大 K、输出 tile 数不足 | Split-K 增加可并行工作 | partial buffer、combine、不同归约顺序 |
| MoE 的 expert 小组 | gather/分组或排序后复用 | 排序、重排、scatter、热门专家与空组 |

以上是工程假设，由实际 trace 与 workload 确认，不能把 dense 模型经验外推所有 MoE。

## 两个可追溯的上游例子

[MLX PR #3018](https://github.com/ml-explore/mlx/pull/3018) 于 2026-01-26 合入 NAX Split-K。它把 K 分区再合并，展示了特定 M5、大 K 范围的收益。讨论明确移除了测试用 `MLX_DISABLE_SPLITK_NAX`，所以不能照抄初始描述中的变量。dispatch 阈值必须从固定版本确认；没有完整本地配置就不引用其数字作预期收益。

[MLX PR #4572](https://github.com/ml-explore/mlx/pull/4572) 于 2026-09-28 合入 `gather_qmm` 改进，作为 MoE 路径可优化的具体线索；该页表格不足以推断本机整模型速度。按允许的参考访问权限查看最终变更，再在相同模型与形状上比较。

## 索引、精度与边界

`gather_qmm` 的索引指向 batch 维展开后的矩阵，scale/bias 的 batch 维必须与权重对应。只有索引确实有序时才设置 `sorted_indices=True`；这个提示并不代替排序。[API 约定](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.gather_qmm.html)

上游 [issue #4632](https://github.com/ml-explore/mlx/issues/4632) 报告了特定 M5 Max/macOS 27、较大非整 tile 行数的有序 gather 错误。这是未在本地复现的作者报告，不能视为所有版本的确定缺陷；使用相关优化前检查状态并覆盖尾部。Split-K 同样需要检查 partial dtype 与合并误差。

验证覆盖小 M、不同 K、非 tile 倍数、重复/空 expert、非连续输入、量化分组和质量指标；测完整 gather→compute→scatter 路径。若一条优化仅适用部分形状，保留经过验证的 fallback，并测试 guard 重叠与遗漏。条件外的错误或慢实现应明确退回调优，不能藏在策略后。

FFN 激活可另读 [SwiGLU 三路比较](mlx-swiglu.md)：库已有 compiled helper，custom 改变舍入，计时有进程排序反转。该运行只测两个已驻留输入的激活，不含 gate/up/down GEMM 或 expert 路由，不能当作 GEMM epilogue、完整 FFN 或 MoE 收益证据。
