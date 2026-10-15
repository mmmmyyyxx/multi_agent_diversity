# Single-Prompt native multi-Parent V12

已按用户确认，将 A/B/C 的公共单 Prompt 内核改为容量3、每轮3槽位的
Parent 池。此报告记录实现与离线验证，不包含该新内核的真实实验结果。

| 环节 | 新行为 |
|---|---|
| Parent 抽取 | 直接实例化正式 `FitnessWeightedSampling`，正概率混合、按 fitness 加权，有放回抽3次，轮首冻结 |
| fitness / 保留 | 完整 Optimize150 正确率；直接调用正式 `BestThreeRetention`；所有完成测评的 Child 参与 Top3，与 Validation 拒绝独立 |
| 候选 | 每个所选 Parent 独立生成1个单块修改；每槽最多4次格式生成；no-op 记为跳过，不强制重试 |
| 诊断 | 运行 Root 每轮诊断一次，三个槽共享；Optimizer 同时看到所选 Parent 在代表题上的真实作答，明确区分 Root 与 Parent |
| 中间门槛 | 三个 Child 全部测 Optimize；仅本轮 Optimize 最高者进入 Validation100，严格多对至少1题才更新运行 Root |
| 收尾 | 直接调用正式 `PerMemberArgmaxAssembly` 的单池 argmax；Validation 不低于运行 Root 才通过；拒绝明确记录，不暗中替换为 Root |
| Test | 只有全部最终提案通过收尾门槛才可进入另行冻结、授权的自动 Test |

正式 Parent 容器要求五成员团队和真实 Vote，不能伪造四个成员来复用容器。
这里复用三个实际策略类，以真实单 Prompt Optimize 分数作有记录的容器适配；
团队 orchestrator、原生策略源码、Memory 和 Group 行为未修改。

新默认仍是每组3重复、Seed42、20机会，共9条；每条最多60个候选槽。
Parent 抽样不加组名、重复编号或路径盐；新 Validation100 的抽样种子也是42。
原始 MATH 三个数据分区不重抽，旧 Seed88 子集不改写。
现有生成接口没有暴露 provider generation seed，不能把算法 Seed42 说成模型输出确定性。
Solver 仅在同一重复、同一阶段、完整请求一致时共享缓存；Teacher 每槽重新调用。

保守训练成功语义及在途调用上限为437940，独立 Test 为10800。
格式耗尽与 no-op 分别记录槽位和机会计数；计费、有限语义重试、无限延迟网络恢复保持显式。
同一 Prompt 已记录的 Validation profile 复用，不通过重测来改变门槛。

新的冻结入口要求干净已提交源码、完整输入哈希和新的精确授权。
旧检查点不能自动恢复成含完整 Parent 池的新协议；历史单 Parent 结果不改写、
不能充当这个新内核的匹配对照。

验证结果与覆盖见 [verification.json](verification.json)，实现文件摘要见
[source_hashes.json](source_hashes.json)。新旧历史证据完整性核对见
[historical_integrity.json](historical_integrity.json)。测试不包含全部历史/私有回放。

分类 current 共1125项：合并复核证据为1123通过、2跳过，无未解决项。
初轮受主机内存压力和验证日志使用 `Path.open` 的问题打断；保留通过记录，
复核全部失败及最新功能模块，再续跑未完成项，没有声称初轮全绿。
最后的登记测试在验证进程按文件内容缓存 `safe_load` 解析，每次仍读当前文件、
深复制结果并执行全部原校验/断言；修改缓存返回值和改文件内容的控制检查通过。
该解析缓存未加入生产代码，不改变模型缓存或实验策略。
生产 compileall、治理/manifest 预检和九条完整 fake smoke 通过，网络尝试为0。

另有用户随后授权的[历史 A/B Test-only 范围](../../docs/design/SINGLE_PROMPT_COMPLETED_AB_TEST_V13.md)：
仅补测六条旧最终 Prompt，不启动这里的新多 Parent 训练，不重跑已完成的 C Test。
该范围使用独立 manifest、授权、账本与最终结果，不能与本报告的离线验证混为一谈。
