# Gradient–Pattern 多方向单 Parent 调度：离线准备

状态：代码与离线验证完成，真实运行资格仍被既有治理问题阻塞；没有真实性能结论，也没有真实 API、历史续跑或
Test 内容访问授权。审计起点为 `ccd97d81825271ee8a5ff9a12a54ad371a77e718`。
最终运行身份须由干净提交后的独立准备清单绑定，不能借用历史授权。

规范见 [调度协议 V1](../../docs/design/SINGLE_PROMPT_PATTERN_SCHEDULING_V1.md)，
配置见 [独立 manifest](../../experiments/manifests/single_prompt_pattern_scheduling_v1.json)。
本报告是准备证据，不是新团队算法的设计权威。

## 三组的真实区别

| 组别 | 每轮诊断 | 最大候选 | 最大接纳 | 接纳规则 |
|---|---|---:|---:|---|
| C0 | 轮初全量错题，选择最大 Pattern | 1 | 1 | 比当前 Validation 至少多对一题 |
| C1 | 同一次诊断的 Top-2 | 2 | 2 | 先决策 A，再基于实际保留 Parent 生成 B；各自严格提升 |
| C2 | 同一次诊断的 Top-2 | 2 | 1 | 同父独立生成完再测评，最高 Validation 胜；平局按 Pattern 顺序 |

C1/C2 复用同一正式 `FullDiagnosisPatternModule`，不复制 Gradient、Cluster、
分区校验、代表采样、解析器或评分器。相同诊断输入会得到相同 Top-2 顺序；
实际轨迹分叉后不保证诊断相同。排序继续使用单 Prompt 的去重支持数及原有
Seed/轮次并列处理，不伪造团队责任。每方向最多三道正式采样代表题。
Teacher 响应不跨组复用，因此即使初始 Parent 相同，实际生成的诊断也可能不同；
这里保证的是相同冻结诊断下的选择规则一致，不是强制各组得到相同 Pattern。

C1 保存轮初诊断来源和每个候选的实际 Parent。第二方向已无当前错题则跳过，
不重诊断、不换第三个 Pattern。否则从剩余支持题正式采样，原始 Pattern ID、
支持集及旧诊断保留，并明确附上当前 Parent 的实际 Optimize 作答。
记录同块覆盖及前一步已修复题的保留/破坏；这些指标不构成额外门槛。

C2 所有生成先于任何候选测评，各候选只能看到自身方向，不合并修改。
V1 工程传输顺序派发，科学上的 Parallel 指同父独立提案，不声称 API 同时执行。
只有一个 Pattern 就只处理一次；没有 Pattern 仅有一个原始 Batch 回退。
格式耗尽、no-op 或目标消失不通过复制方向补齐候选。

用户已再次确认仅保留当前非空三块、单块替换和解析检查，**不恢复内容 Guard**。
no-op 跳过测评，不强制合法性重试。结构失败首次加最多三次重试。

## AgentGrad 与本地适配

[AgentGrad 官方论文 v2，第4.3/4.4节与算法1](https://arxiv.org/html/2609.08572v2)
描述语义聚类后的方向调度和累积更新，并使用当前语义批次提升后再检查
Validation 的接纳流程。本实验 C1 只借鉴顺序累积调度：使用仓库原有诊断，
不增加 Intervention、Hint、伪标签或语义批次接纳门槛，只处理 Top-2。
因此不能称为 Original AgentGrad 复现，也不能从论文结果推断此适配有效。

## 数据、重复与成本边界

保持原 Optimize150 和正式 Test300 成员，新的共同 Validation100 抽样种子42。
每组3次、全部算法与可配置抽样基础种子42、每次20外层机会。重复编号用于
缓存隔离，不作为随机种子盐；固定种子不保证供应商生成完全确定。
不依据此前六题推理审计改动本次成员或评分。
正式 Test300 在历史工作中已有暴露；新运行仍须按阶段门禁封存并单独授权，
但其结果不能被称为首次盲测或新独立确认。

| 范围 | 外层机会 | 最大候选 | 成功语义及在途调用上限 |
|---|---:|---:|---:|
| C0（三次） | 60 | 60 | 90,420 |
| C1（三次） | 60 | 120 | 150,660 |
| C2（三次） | 60 | 120 | 150,660 |
| 训练合计（含自适应 Validation） | 180 | 300 | 391,740 |
| 最终 Test（九个最终 Prompt） | — | — | 10,800 |

单条训练上限为 `250×4 + 20×[槽位数×(250×4+4)+453]`，
453 为全量最多150题各3次 Gradient 加3次 Cluster，每轮只发生一次。
这些是保守执行上限，不是预期账单；缺方向、no-op、格式耗尽及缓存可减少调用。
累计 Token 和失败网络物理调用不设硬上限；网络失败逐次记账，沿用冻结退避。
没有真实 API 数据，不能给出可信的实际 Token 或完成时间预测。

Solver32/Gradient8、Test Solver16，现有模型、SDK2.37.0、解码和输出容量不变。
只允许同重复、同阶段、完整请求相同的 Solver 终端观测跨组/轮/槽位复用；
Teacher 每次真实调用。新尝试不导入旧缓存、响应、检查点、授权或成本。
共享缓存成本归首次派发单元，组别执行顺序会影响各组物理成本归属。

固定20轮与固定已测候选预算分别报告。C1 逐子步记录真实保留状态；C2 必须等
整轮候选测完才知道胜者，不能在一个候选预算点提前使用第二候选结果。
曲线只有已观测 Optimize/自适应 Validation，没有中间 Test。相同候选数不等于
相同诊断次数、Token、API 或时间成本，也不能据最终 Accuracy 单独宣称效率。

## 变更及可比性审计

新增隔离的方向调度、反馈投影、逐槽工件、完整边界恢复审计和预算曲线模块。
共享生成器只新增显式启用的槽位 ID 与旧诊断来源上下文；旧调用默认参数保持。
共享治理新增严格 schema 路由，拒绝不匹配的准备、授权、账本和收据。
独立配置不使用 `multi_parent.py`，未修改正式五人团队的搜索、责任、记忆、
Solver、评分、接纳或当前实验身份，也未覆盖历史默认单 Prompt 配置。

历史 C 的 Guard/评分/执行版本与本次并不完全相同，不能充当严格匹配 C0。
未来获授权后应在同一新冻结版本下运行全部三组；本次准备不会重跑历史结果。
按评分记录的修复、覆盖差异和词面变化不证明 Pattern 诊断真实或同类错误泛化；
参见 [诊断与迁移审计](../pattern_diagnosis_transfer_audit_20261019/README.md)。

## 验证与运行入口

契约、正式实现哈希及成员摘要见 [contract_audit.json](contract_audit.json)，
验证证据见 [verification.json](verification.json)。所有测试通过离线网络 Guard，
假模型端到端仅使用合成数据。完整 current 套件与历史/私有回放并非同一范围。
完整 current 回归 1205 通过、2 跳过；最终新增功能套件37通过，信息流重查6通过。
这些检查存在覆盖重叠，不能相加为独立测试数。全面回归开始后补充的统计分支
及新增用例由最终功能套件验证；没有宣称全部历史回放通过。

完整实验治理预检有两条已复现的基线阻塞：
`FAIL-ABLATION-PARTIAL-FORMAL-ADAPTER-CONFORMANCE` 和
`FAIL-ABLATION-PATTERN-DIAGNOSIS-UNIVERSE` 的 `evidence_level=source_verified`
不在当前校验枚举内。失败登记文件与审计起点相同，本次不改写历史分类或放宽校验。
在解决此治理阻塞并获得具体 scope 授权前，不能宣称具备真实运行资格。
仓库元数据审计发现此前引用的诊断报告未注册，已仅补齐该报告的发布记录和
来源边，原报告及其证据保持不变。

首次完整测试收集因环境中的 `gmpy2` DLL 搜索失败而中断；验证使用同一
仓库运行时，并仅为测试进程添加现有 Conda DLL 目录，不改包版本或数学后端。

规划与合成演练使用 `scripts/ablations/study.py --config
experiments/ablations/pattern_directions/default.json --plan/--smoke`。
干净提交后用 `scripts/ablations/prepare.py --pattern-directions --destination ...`
建立新的冻结准备；初始 `approved=false`、`READY_TO_RUN=false`。
唯一真实入口仍为 `scripts/run_experiment.py --prep ... --run-root ... --execute`。
执行前须明确授权准确源码、scope、训练/自适应 Validation/最终 Test、角色和预算，
然后才能设置 readiness；全部训练与终端审计完成前不得打开真实 Test。
没有创建或修改定时监控，没有启动实验，没有推送。
