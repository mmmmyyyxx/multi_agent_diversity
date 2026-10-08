# 实际信息流审计

判定词限定为 `CONFIRMED DEFECT`、`INTENTIONAL DESIGN LIMITATION`、`SCIENTIFIC RISK`、`NOT REPRODUCED`。缺失能力不自动意味着旧实现违背旧契约。审计读取了旧执行 SHA 的关键代码、已封存的七机会取证及当前生产组合，并检查合成 provider 的实际序列化请求，而非只检查注释。

| 字段 / 边界 | 来源与实际消费者 | 已观察缺失、丢失或误名 | 判定与科学意义 | V2.3 校正 / 验证 |
|---|---|---|---|---|
| problem / final reference / worked solution | Canonical MATH → typed example → Solver / evaluator / Gradient | 旧 Gradient 只有题面、final reference；worked solution 留在数据中 | CONFIRMED DEFECT：本任务要求的参考过程没有传到优化诊断 | Worked solution 只随 Optimize evaluation example 进入独立 Gradient，和 Solver 轨迹分开标记；ID/split tamper 在 provider 前拒绝 |
| dataset type / level | Canonical source → private competence audit | 没有细粒度运算标签 | INTENTIONAL DESIGN LIMITATION：不能据此发明数学能力专长 | 保留可靠 type/level 私有 metadata；可见记忆只陈述实测覆盖计数 |
| member / procedure / input / realization IDs | Binary state → visible packet → Gradient / Layer1 | 同一文本 prompt 仍需绑定实际成员 realization | SCIENTIFIC RISK：错配会制造错误反馈 | V6 原有 provenance 校验复用；检查实际序列化请求和来源 hash |
| ordinary Solver content | Provider → private profile → bounded visible solution | 旧 Pilot 输出接口没有要求数学步骤；当前 V6 已修复 | INTENTIONAL DESIGN LIMITATION（历史）；不能把当前接口说成旧实测轨迹 | 保留 V6 完整私有响应、4096 字符投影及 missing / boundary / truncation 状态 |
| final payload / validity / correctness | Existing parser and math equivalence → scoring / feedback | invalid 永远不是正确；不能用轨迹猜答案 | NOT REPRODUCED：未发现本轮 scorer 或 V3 gate bug | 无 parser 放宽、无答案 fallback；valid-wrong 与 invalid 分列，旧格式边界测试复用 |
| reference trajectory vs observed trajectory | Dataset worked solution / actual Solver response → Gradient | 参考过程不能冒充模型自己的步骤 | SCIENTIFIC RISK：参照解不证明隐藏根因 | 独立 source 标签、split、ID、长度和截断；合成相同 final answers 的不同可见操作能被分辨，未声称真实诊断质量 |
| per-example Gradient output | Independent Gradient calls → semantic clustering | 旧单 string 强制 uncertain failure 也输出修复行为 | SCIENTIFIC RISK：可能将不确定观察升级成命令 | 紧凑 observation / diagnosis / correction / effect；`UNCERTAIN` 为合法无动作结果；每错题独立接受一次 |
| semantic gradient partition | Abstract individual corrections → cluster → same raw Pattern responsibility | Cluster 不接收原始 Solver 历史和团队元数据 | INTENTIONAL DESIGN LIMITATION，层次分工正确 | 仍只聚类 actionable corrections；没有语义混批诊断，也没有新 Pattern controller |
| WHO / raw responsibility / allocation | Layer2 diagnosis → target selection / audit | 原 mutation selected_pattern 含 raw responsibility_value；完整 optimization_context 本来已过滤 | INTENTIONAL DESIGN LIMITATION（过滤）；不应将过滤本身当 bug | 保留 WHO 与团队责任于 Layer2；新 mutation wire 只传 selected hypothesis，不传排名、failure counts 或目标分数 |
| selected generalized gradient | Selected Pattern → Layer1 | 原指令要求执行唯一 repair objective，缺少安全 abstain | SCIENTIFIC RISK：局部假设可能被广泛强制使用 | 视为修复假设；小范围 standalone edit 或 `NO_SAFE_EDIT`；重复/无动作不导出 |
| Mutation membership | Pattern support + current-correct pool → mutation examples | 旧实际 panel 为三条 repair 加同一 anchor，共四题 | SCIENTIFIC RISK：小机制收益不能代表全面保留 | 最多两条 support repair 加保留样本，三条生成例子；精确 membership 冻结于机会 |
| SearchValidation membership | EvidenceView → local evaluator / selection | 旧与 Mutation / Probe 完全同集合 | CONFIRMED DEFECT：命名为 validation 却没有独立例子迁移检查 | 三条 disjoint Optimize validation；parent/child 同集合；生成后测量，只有既往计数进入下一迭代 |
| preservation anchors | Committed correct records → local validation | 旧 deterministic min 在七机会使用同一题；21 个旧正确未在 panel 覆盖 | SCIENTIFIC RISK：已有 preservation 机制但覆盖不足 | 种子/member shuffle、每成员机会旋转两位；候选 gains 不更新 pool，commit 后取新实测 pool |
| TeamProbe / broad_delta | Fixed-peer evaluator → unchanged promotion | 旧 Probe target 复用 panel/cache；broad_delta 实为 target_delta | CONFIRMED DEFECT（独立性/误名）；不能解读成新样本复现 | Probe 另取三题并记录 scope；promotion 公式不变；仍不把 broad_delta 名称解读成更多例子 |
| Full / deployment floor | Whole Optimize → V3 transition → winner gate / commit | 两个旧 Full 新增少于丢失，拒绝正确；未有旧 commit | NOT REPRODUCED：七次零 commit 不证明 gate 实现错误 | Full authority、初始 floor、Vote nonregression、target-or-team progress、单成员与固定 peers 均保留 |
| root / local parent / deployed parent | Local search lineage / Binary store → local and Full effects | 局部父节点可变但部署团队仍无 commit | SCIENTIFIC RISK：不同参照可能混淆收益 | 同时保存 actual-parent validation、root-relative local、Full deployed-parent scope 与 hash |
| initial competence Memory | Measured initialization → rolling Memory | 旧 LLM Memory 在开始为空 | INTENTIONAL DESIGN LIMITATION：原 policy 不提供能力 bootstrap | 从已有 Optimize profiles 生成五条私有记录，零新增 Solver/LLM，原始/当前/新增/丢失覆盖分开 |
| intended change vs actual diff | Mutation metadata / deterministic parent-child comparison → edit Memory | 旧 closed action labels 不能重建实际修改，summary 只是意图 | INTENTIONAL DESIGN LIMITATION：不是完整编辑后果学习 | 每个已评估编辑保存 lossless private diff、意图、support/hypothesis、四种 scope 后果和状态；合成故意不一致测试 |
| rejected Full experience | Full evaluator → opportunity Memory transaction → next same-member search | 原一般 shared risk 缺少候选自身 Full 获益和 collateral loss 的完整私有关联 | CONFIRMED DEFECT：本任务所需学习证据未接通 | 零 commit 也保存 Full-specific fixed/broken/retained；容量败者和 operational fault 不获得虚构 Full 状态 |
| optimizer-only dependencies | Generated procedure → Solver contract guard | 旧实测 mutation 出现引用不可见上下文的流程 | SCIENTIFIC RISK：合法措辞不等于可执行输入 | Known-field / ground-truth / lookup / long copied-fragment 检查；允许一般数学检查，明确词法边界 |
| private data / shared risk / protected splits | Current ports and governance → runtime / reporting | 公开不能包含私有题目、答案、参考解、prompt 或 provider payload | SCIENTIFIC RISK：改善反馈不能突破数据边界 | 私有原文留 ignored；公开 hashes/categories/counts；Optimizer-only 数据不进 Solver，Validation/Test 真实访问为零 |
| evaluation stability / optimizer quality | Real realizations → scientific interpretation | 没有新的真实配对测量，无法区分模型随机性和修改因果 | NOT REPRODUCED：本轮不建立这些因果结论 | 下一比较重新测 baseline；旧 22/60、P2 与 V6 baseline 不可混用 |

旧七机会的真实事实仍引用 [原零 API failure audit](../math_v2_2_optimizer_failure_audit_v1_20261008/README.md)：实际四题 panel、重复 anchor、panel=Probe、33 合格已评分、3 个 root-positive、两个 Full 丢失 11 次。这些事实没有因 V2.3 实现而改写。
