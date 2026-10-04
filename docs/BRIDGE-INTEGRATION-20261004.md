# 计算与通信计划进入 PyTorch bridge

本轮目标：普通 PyTorch/HPU Graph 应用只换 bridge wheel，取得外置原生执行器的
性能；不修改权重、精度、计算 kernel，不再依靠额外的 Python/C++ 执行器。
目前已完成小图、分布式最小链和两层模型接线；70 层性能验收正在执行。

## 实现及收益来源

源码在 `tracycam/gaudi-pytorch-bridge` 的 `feat/native-graph-runtime` 分支。
compute/collective 功能候选 `6169def46`；加入公共输入 replay 修复后的候选
`413e535ce`。bridge 后续文档提交不改变已编译 wheel 的身份。

1. 编译结果由 `HabanaLaunchOpPT` 的正式 fresh/cache 接点导出。公共 launcher
   不再认识捕获状态，降低执行层与 Lazy 层耦合。
2. 捕获计划拥有计算 recipe、typed collective、冻结参数、抽象绑定及 Tensor Storage。
   同步重放跳过逐子图 Lazy cached-execute。Storage 独立持有，boundary 按槽验证；
   内容可变，但不能把同图的另一个 Tensor 偷换到当前槽。
3. 通信仅先支持 FP32/BF16 all-reduce/all-gather-out。整计划进入已有 HCCL 队列一次，
   队列内逐条 SDK 提交，省掉每次 collective 的独立 job/promise/future；没有设备端
   单 launch 的声明，原地址锁、事件和资源管理保留。
4. 公共 `replay_with_inputs` 原先遇普通 HPU Tensor 会不做任何计算直接返回，
   是 API bug。现在通过既有 V3 adapter，补齐提交前 input count/shape 检查与 reset
   元数据清理。重绑是正常回退路径，尚未宣称快路径支持。
5. serving 增加明确 `executor=pytorch`，计算算子注册保留，外置 preload、executor
   启动及特殊输入/采样传输关闭。原 worker 没有原生传输对象时使用 PyTorch superclass。
   A/C 两臂使用同一冻结应用、同一 manifest 计算策略、同一权重和 TPC 库。

实现细节见 bridge 的 `docs/native_graph/IMPLEMENTATION.zh-CN.md`。

## 已完成的验证

| 范围 | 结果 | 限定 |
|---|---|---|
| 构建 | 全部 `-j 32`；最终 wheel `1.24.1+git413e535ce` | 独立 build/venv，不修改共享 SDK 或 Torch |
| CPU C++ Storage/queue | 6 项通过，构建 `6169def46` | 真实 Storage 重绑、alias、extent；队列异常传播和嵌套提交 |
| 配置/serving/placement | 31 项通过 | 清洁普通启动分支和原路径兼容 |
| 普通 wrapper | FP32/BF16 × M1/2/8/64/512 × 8 变化输入，最终 wheel 通过 | 小图，不是模型 TPS |
| 输入 adapter | FP32/BF16 各 8 个新分配 + 2 个 reset/recapture 新输入 + 2 个非法输入检查 | native hits=0，明确属于回退 adapter |
| stream/view/churn/reset | 最终 wheel 各 16 轮通过 | reset 只在末轮做一次无显式同步释放，不宣称完整在途压力测试 |
| TP2 混合链 | FP32/BF16，各 all-reduce/all-gather/mixed-mode；共 96 个变输入检查通过 | rank0 原生、rank1 V3 回退；harness 外部诊断一致性检查不是 runtime 自动协商 |
| 两层 TP8 | 8 rank 每个 11 命令/5 collective/45 native replay，32 token 完全一致 | 输出等价；部分层不验语言质量或整模型性能 |

普通 wrapper 的早期候选 `4e46f2984` 与 upstream fresh-process ABBA：完整小图
重放约 202–210 → 147–156 µs，改善 1.35–1.38 倍。含 Python wrapper hash/copy、
bridge 提交及区间结束等待，不能称纯 GEMM 时间或乘到整模型 TPS 上。

两层模型旧 compute-only 候选曾全部回退；mixed 候选已真实命中。变化输入与公开
下游 Tensor 消费必须一起验收，不能只看 replay 计数。

## 70 层实验

冻结应用 `c54d954ae033d367af520f363ca65840144055ea`，权重 MiMo-V2.6-Pro-RL，
70 层、TP8、无 MTP、4K 上下文，B1/B2 各 128 token、2 轮。
上游 bridge `5176b1b2d6865608514e83a2b1fd66459920676e` 对比候选 `413e535ce`。
计划 fresh-process A/C/C/A，模型不随 arm 改代码。resident interval 与包含 prefill 的
整个 generate wall 都保留；snapshot 在计时结束后取。结果完成后追加本节。

`tools.validation.executor.compare_bridge_model` 对比完成状态、同应用配置、各请求输出、
全部 rank 的 native coverage，并分别报告 resident/total TPS。部分层明确
`full_model_qualified=false`，无 70 层或覆盖不全不得宣称达到模型目标。

## 本轮失败及修复

- 双卡早期进程设备选择发生在 Torch autoload 之后；改为新 subprocess 在 import 前
  设置 module/rank，并保留完整 SDK visible mask、按 PCI/NUMA 绑核和独立 rank 日志。
- marked inputs 配 bare replay 的错误诊断触发旧 bridge 的空 Lazy Tensor；GDB 留档。
  正确 V3 对照运行通过。真实 `replay_with_inputs` HPU 输入 no-op 单独修复。
- 初次完整模型命令的 `--out` 被 argparse 缩写成 supervisor 的 `--output-root`，
  输出目录相撞、未执行模型。全部转发 CLI 禁用 abbreviation，supervisor 也拒绝复用
  已存在 exit 记录的 case。早期重复同 ID 的启动失败已明确记录，不混入性能结果。
- 两个早期源码声明字段填错已留 correction JSON；现在启动前验证冻结 export 身份。
- 旧探针的 HPU `set_` 操作不受该 SDK 支持；不冒充 Tensor owner gate 已上卡通过。
  CPU 真实 Storage 测试覆盖其逻辑，但 HPU 动态重绑验收仍需扩展。

## 尚欠的工作与弃用条件

先拿完整模型同应用效果与覆盖，低于 90 TPS 时按实际 CPU/SDK/设备时间定位差距。
不能借旧 executor 模型专用 RPC/采样/input staging 加速掩盖 bridge 差距。
再做分布式 in-flight reset/communicator 释放、异常及不支持 collective 的完整矩阵。
动态 shape、训练/RNG、dry-run materialization、快速输入重绑仍是明确回退范围。

旧 native executor 与外置 native_graph 暂只留作参考/验收对照。满足完整普通应用
性能、正确性和生命周期矩阵后，才从 serving 中删除并归档；此时生产只保留 bridge
执行能力与独立计算 kernel 库。MTP/prefill 在此基础上继续，而不是再搭第三套调度器。

## 资产与 Git

用户的“资产和成果必须留在本地”作为持续约束执行。
本地资产目录为 `gaudi-kernels/artifacts/builds/bridge-integration-20261004/`：源码 Git bundle、
冻结应用 tar、最终 wheel、SHA、构建/探针/model 原始日志及失败均保留。
早期资产不覆盖。wheel SHA256：
`59a0013768a3b203693bf4569e1234728b9fa96958671ee0f6cc33d51049834a`。
公开 Git 只上传代码和说明，不上传权重、私有业务数据及大二进制实验资产。
