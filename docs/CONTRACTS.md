> Historical kernel contract inventory. See [current status](STATUS.md) for the
> integrated model boundary; entries below are shape-specific, not general guarantees.

# 格式、算术合同和执行引擎分开管理

W 是权重位宽，A 是激活的算术精度；
不要把 BF16 MME 当成权重存储格式，也不要因为 Python 输入张量是 BF16，
就忽略 kernel 内部已经发生的 FP8 激活量化。

| 权重格式与 scale | 激活合同 | 小 M 候选 | 大 M 候选 | 现状 |
|---|---|---|---|---|
| FP8，per-tensor/per-channel 等可分离 scale | W8A8 | native FP8 MME 或 TPC | native FP8 MME + FP32 累加 | 已有原生图与公开框架接口；精确测试形状见FP8报告 |
| 同上 | W8A16 | SRAM解码/MME，融合TPC待优化 | SRAM 解码 + BF16 MME | 已实现原生与框架路径、原A16和FP32 scale收尾；SRAM逐图验收 |
| FP8，FP32 128×128 K-block scale | W8A8 | native FP8 + SRAM 部分和/归约 | K-block 流式缩放累加候选 | 本 QKV 小 M 有实测；大 M 通用高效方案仍缺 |
| 同上 | W8A16 | TPC 或 SRAM BF16 MME | SRAM BF16 MME，按几何分块 | 已测 QKV；框架图内接入未完成 |
| BF16 | W16A16 | TPC GEMV 与 MME 比较 | 厂商 BF16 MME + FP32 累加 | 110配置通过；选定MME/SRAM/尾行计划，TPC候选未胜出 |
| MXFP4，E2M1 + 每32个K元素共享E8M0 | W4A16 | 融合 TPC GEMV | 按专家 grouped GEMM，TPC解码/SRAM/BF16 MME | 新N256原生及静态框架组已测；HistoricalN512设备分桶T≤513已测，模型默认接线待完成；E8M0仅2..252 |
| MXFP4，同上 | W4A8 | 只在显式允许激活量化时研究 | FP8 MME + 对应块缩放/归约 | 可选实验，不能自动替换 W4A16 |

FP8 MME 执行原生 FP8 操作数。把 BF16 激活转换到 FP8 会改变激活合同；若要
保持 A16 信息，必须另有经过证明的分解/补偿实现，并计入其额外计算与搬运。
“FP8 MME W8A16”不能仅靠改名字成立。

BF16 MME 可以对已经量化的值执行 W8A8 数值合同，但通常只应作为对照或有
明确精度/硬件理由的备选。无需为了凑齐排列组合而让每条主路径都先量化、
再反量化。FP32 是累加精度，不代表应该让权重、激活都扩成 FP32 输入 MME。

MXFP8 是每32元素共享 E8M0 的另一个格式，不能与 FP32 128×128 scale 互换。
标准定义参考 [OCP MX specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)。

## 统一接口需要携带的合同

未来的 `linear`/`grouped_linear` 接口必须明确：weight encoding、原始/准备后
布局版本、scale dtype/分组轴/块大小、是否已做 native 格式适配、逻辑 N/K、
激活量化策略、累加/归约/输出 dtype、bias 的位置、TP 分片与输出归约语义。
W/A 位宽二字不足以描述这些信息。加载时范围适配不能重复执行。

prefill/decode/verify 共享线性代数入口；差别进入 M、任务表和 attention 的
请求/KV 元数据。接口通用不等于所有形状共用一个 kernel 或一个 graph。

## 按实测选择计划

计划键包含 `(device, runtime ABI, format/layout, scale policy, A policy,
M/N/K bucket, output dtype, bias/epilogue, TP partition, graph mode)`。
容量条件先淘汰不合法方案，再比较同一完整算术合同下的完整算子时间。不能
把 RMW 的 M=25 容量边界当速度阈值，也不能用单个 MiMo 形状概括全部模型。

小 M 优先看原始权重字节/时间、TPC issue/访存、填充排空和固定成本；大 M
优先看有效 FLOPs、MME 几何空位、权重跨 M 的复用、SRAM 与解码重叠。
很小或严重不整齐的矩阵未必能打满带宽/算力，需报告形状对应的可达上限。

验收至少包括完整 MAC 误差、近零/相消/极值、尾块、空任务、bias、TP 分片、
原位宽存储、编译布局、运行期拷贝/重读，以及真实模型质量。编译图“逻辑
读一次”不等于物理 HBM 事务严格1.000×；当前后者仍有证据缺口。
