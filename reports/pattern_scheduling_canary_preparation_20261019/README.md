# C0/C1/C2 独立 Canary 准备

这是离线准备记录，**不是已完成 Canary 或正式实验的结果**。没有真实 API、
真实 Solver/Validation 评估或 Test 内容访问，也未授权新的推送。
前置治理修复见
[兼容性报告](../source_verified_failure_registry_compatibility_20261019/README.md)；
两条历史失败记录及证据分类未修改，完整治理预检恢复通过。

## 具体执行范围

规范：[Canary V1](../../docs/design/SINGLE_PROMPT_PATTERN_CANARY_V1.md)。
独立 [Manifest](../../experiments/manifests/single_prompt_pattern_canary_v1.json)
复用现有单 Parent C0/C1/C2 内核，各1次、1轮，Seed42。
完整 Optimize150、Validation100（抽样Seed42）、模型、SDK、输出容量、
Gradient/Cluster、代表题、单块结构检查、严格提升门槛和缓存身份不变。
C2 是同父候选择优，真实生成请求仍顺序派发，不声称实现 API 并发加速。

| 项目 | Canary 冻结设置 |
|---|---|
| 单元 | C0、C1、C2，最多5个候选槽 |
| Solver / Teacher | gpt-4o-mini / qwen3.7-flash |
| 训练并发 | Solver32、Gradient8 |
| 成功语义响应及在途上限 | 9,379（2,457 + 3,461 + 3,461） |
| 累计已记账及预留Token硬上限 | 3,000,000 |
| Test | 禁止；授权必须为false，上限0 |
| 旧实验状态/响应/缓存/成本 | 不导入 |
| Solver共享 | 同重复编号、同阶段、完整请求一致时共享 |

Token上限由既有 broker 在每次派发前检查，包含并发预留、网络失败的保守
计费；与供应商实际收费不作等同。语义调用上限排除网络失败物理调用。
触及硬资源上限保存证据并结束，不自动扩大预算或新开进程。
正式实验的无限累计Token设置不因此改动。

真实 Pattern 数量不足、C1 首候选未保留、第二方向已被修复等情况会减少
实际覆盖。分别记录方向数、生成/测评槽数和最新Parent分支是否执行。
不能为了得到两方向覆盖而造Pattern、强制接纳或自动重跑。
Canary完成仅表示已执行路径的运行与审计情况，不证明诊断准确、泛化或收益。
Canary的状态和结果不能并入正式九条轨迹。

## 实现和验证边界

新增独立 Canary manifest/schema、冻结/验证/执行包装和覆盖摘要。
统一入口显式路由该schema；Test授权即便随 `approved=false` 提交也拒绝。
共享 execution plan 现在显示 scope 的真实Token上限；原有无限上限仍为null。
调度算法、真实Gradient/聚类、评分及候选选择没有修改。

新检查与复用范围记录于 [verification.json](verification.json)。
首轮定向回归108项通过；最终Canary与共享入口回归70项通过，两轮存在
重叠，不将它们相加当作独立用例总数。生产编译、完整治理与Manifest预检通过。
复用此前完整 current 的1205通过、2跳过，仅覆盖未修改的科学/运行代码；
不冒充本次重新执行的完整套件。受影响的调度、冻结授权、共享入口与治理
另行在零凭证网络Guard下验证，包括三单元完整假Provider训练、账本、
回执、缓存和终端封存。假Reader的Test入口直接抛错，确认训练不会读取Test。
历史/私有资产回放不在本次检查范围。

## 待执行授权

干净提交后，生成新的独立source SHA、scope、handoff及执行计划；
其默认 `approved=false`、`READY_TO_RUN=false`、`test_access=false`。
离线预检不能代替真实链路验证或 API 授权。
依据仓库AGENTS.md第8节，在精确范围、源码和预算冻结后申请本次授权。
正式九条优化和最终Test另行授权；历史Test已有暴露，不称新的独立盲测。
