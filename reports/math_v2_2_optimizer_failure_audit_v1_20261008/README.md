# V2.2 A4：42 个 proposal 的零 API optimizer failure audit

本次证据最明确地支持 **局部 repair 检查与 Full preservation 覆盖失配**，并观察到 **optimizer 上下文被写进 Solver 决策流程** 的具体 mutation 风险。未观察到前四 export 截掉净收益更高候选，也未观察到正局部收益无法转成正 Probe 收益。两个 Full 没有实现 P2 式 additive repair；第三个同分 positive Probe 没有 Full，其结果未知。不能由此断言 optimizer 没有生成 additive repair 的能力。

这是对已经停止的单次尝试的回溯审计，不是新的 A4 实验、方法修订、泛化评估或因果对照。新增 API、Shadow、Validation、Test 调用均为 **0**；V2.2、generation 上限、模型、selection 和 transition 均未修改。原授权仍关闭。

## 证据范围与一个重要前提修正

范围为 `math_v2_2_gradient_pattern_seed81_pilot_v2` 前 7 个完成机会、42 个 proposals、27 个实际 TeamProbe 和 2 个实际 Full。第 8 个未完成机会不进入分析。来源为 [已发布的停止报告](../math_v2_2_gradient_pattern_seed81_pilot_v2_execution_20261008/README.md)，执行源为 `6731aa3f345eff7feac12140b4892f4e07291d17`，父发布提交为 `0de6a6d08774948d8fbac010eb4a953799dcfa74`。

**这 7 次实际 panel 都只有 4 题，不是 6 题。** 6 是配置容量上限。实际组成均为 3 条 repair 证据加 1 条 root-correct anchor；初始 panel 正确数都是 1。每个机会评估 root 和每个合格候选时使用同一 panel。七个机会的正确 anchor 还是同一道题：局部检查合计只覆盖原有 22 道正确题中的 1 道，另外 21 道不在 panel。

七次的目标成员为 0、1、3、2、4、0、1；七个 outer parent 字节相同，五个成员初始均为 22/60，Vote/Oracle 均为 22/60，commit 为 0。它们是对同一初始团队进行的七次不同成员局部尝试，不是七次成功部署后的 sequential team evolution。目标成员、fresh 梯度/Pattern 和失败 Memory 随机会变化；机会内 local incumbent 也可以变化，不能与部署团队混用。

## 局部收益与排名

42 个 proposal 中，33 个契约合格并完成 panel 评分，9 个在评分前被契约拒绝。33 个中相对 **机会初始 root**：3 个正收益、25 个中性、5 个负收益。42 次生成的正收益率为 3/42，合格候选的正收益率为 3/33。相对 **生成时 local incumbent** 只有 1 个严格改善：O1G2。O1G4、O1G6 相对 root 各 +1，但相对已经改善的 O1G2 均为 0。

3 个正收益候选都是局部新增 1、丢失 0，而且修复的是同一道题：局部新增覆盖去重后只有 1 道。5 个负收益候选都是新增 0、丢失那条唯一 anchor。全部合格候选合计新增 3 次、丢失 5 次。这些是重复候选的事件次数，不是题目去重后的覆盖。

实际 export 按 root-relative newly-fixed 降序、newly-broken 升序、正确数降序、generation 升序，最多四个。逐机会重放与记录完全一致。6 个合格但未 export 的候选中，没有任何一个净收益严格高于已 export 的候选。不能支持“经常截掉更好候选”的判断。排序中的 fixed 优先于 net 的潜在风险属于静态设计问题，本批候选没有形成能显示该风险的交叉排序。

O1 export 为 G2、G4、G6、G1。三个正候选全部进入 Probe，所有 Probe 的正指标相同：target/responsibility/broad 各 +1，Vote 和 team-net 均为 0。固定 promotion 的最终 candidate-id 降序 tie-break 选择 G4、G6；G2 被同分限额截在 Full 外。这里丢掉的是 **同分候选**，不是已测更好候选。G2 较短且没有显式外部上下文依赖，值得记录为未测反事实，不能据此赋予它 Full 优势。

## Prompt mutation 与 Gradient → Prompt

人工阅读了 40 条成功解析的候选文本，以及 2 条无效 envelope 的原始生成内容。33 条合格候选都是 root/local-parent 的 replacement；没有合格 append-only。文本替换本身不是失败证据：历史 P2 同样没有保留初始 18 字符原文。

按本报告明确的单人审阅口径，33 条合格候选中，16 条无可用性限定地引用 Solver 不拥有的 optimizer 上下文，例如 reference、过去 prediction、代表性失败轨迹、selected gradient、retrieved memory 或 measured-loss 历史；8 条保留明确限定可用性的 reference-value 核对分支；9 条没有这种显式依赖。全部 42 条分别为 21、9、12。标记表示可见输入假设，不表示整条 prompt 必然无法求解；分类不是穷尽的语义安全证明，也没有测量多审阅者一致性。

这是 **控制器上下文与可执行 Solver 流程的语义混淆**，不是实际把 gold、gradient 或 Memory 输入了 Solver。当前 Solver 只收到不可变接口、mutable procedure 和公开题面。尤其 O1G1 将读取 reference/prediction、失败 gradient 和经验作为求解步骤，却仍通过现有有限语义契约；不能靠这些输入完成它写出的诊断循环。

七个 selected generalized gradient 的可见指令分别围绕代回验证、reference/替代方法验证、中间计算及定义域检查、定义到公式映射、结构与假设映射、几何一致性、算术与代数交叉验证。生成内容都能看到对应操作的文字表达。因此“完全没写出 gradient 指令”不是本批的主要症状。但是，写出指令不等于 Solver 执行，宽泛 verification 也不构成潜在根因的证据；代表轨迹仍只有最多三条。部分候选还把实例结构或 optimizer 字段带进通用流程，15/42 有人工标记的具体实例展开痕迹。

无效候选中还看到了把几何特例外推为一般公式、把 valid 与 correct 混为 preservation 条件等语义问题；这些已在 envelope/契约处被隔离，没有进入本次 Full，不能用它们解释两个 Full 的丢失。契约拒绝共 9 次：无效结构 2、外部输出接口修改 6、fixed-answer-payload 分类 1；分类名称是原 guard 的观测，不额外宣称检测全面或每次潜在意图已知。

## 两个 Full 的 parent → child 人工细读

两个候选的 outer root 都是同一 18 字符极简 procedure，生成时 local parent 都是 993 字符的 O1G2。Full G4、G6 是完整工作流替换，分别为 1,500、1,321 字符；不是在已部署团队上追加修复。它们使用同一个 selected generalized gradient，核心意图是将候选值代回原式或约束验证。

| 候选 | 相对 local parent 的主要变化（脱敏语义摘要） | 与 selected gradient 的关系 | 可见风险 |
| --- | --- | --- | --- |
| O1G4 | 扩展二次方程、中间函数计算、概率建模分支，增加 preservation 段及 reference/memory 条件核对 | 保留代回与约束检查，但扩展为跨题型全局流程 | 重复 verification；某些场景偏向数值计算而弱化符号简化；提及未提供的 reference/memory |
| O1G6 | 强化方程标准形、分支/对称性检查和 reference-value 核对 | 保留代回与约束检查，新增机制外限制 | 更长的通用工作流；可选 reference 并未实际供给；没有测到全面 preservation |

两者不能简单归类为“完全不合理的 prompt”，其中不少数学检查是合理的。能成立的观察是：一次小 panel 的成功被伴随全局流程替换，随后 Full 出现跨题丢失。现有短答案记录没有可见 Solver reasoning，**不能证明它们在哪一题误用了哪条规则，也不能证明长度、数值偏好或 reference 分支造成了丢失**。完整私有 diff 与原 generalized gradient 留在忽略的审计证据中；公开仅保留哈希、长度、类别和结果，遵守仓库脱敏规则。

| 候选 | Local 正确/4 | Probe target delta | Full 正确/60 | 保留旧正确 | 丢失旧正确 | 新增正确 | Vote | Oracle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| O1G4 | 2 | +1 | 18 | 16 | 6 | 2 | 22 | 24 |
| O1G6 | 2 | +1 | 19 | 17 | 5 | 2 | 22 | 24 |

两者均保住唯一 local anchor，11 次旧正确丢失均发生在 panel 之外。Full 准入正确拒绝了低于 22 的候选。未发现本次 V2.2 gate 的实现错误；但本批没有可准入 winner，因此也不验证 after-commit 路径。

## Probe 与 additive repair 的结论

相对 root 的三个 local positive 均变为 target-positive Probe（3/3），相对 incumbent 的一个 strict positive 也变为 positive Probe（1/1）。没有观察到“positive 在 Probe 消失”；本次转化失败发生在 **Probe → Full 能力保留**，以及成员覆盖新增未转成五人 plurality 改善。后六个机会主要在生成阶段没有得到任何局部 repair 收益，不能把这些 neutral 候选算成 Probe 错杀。

进一步核对发现，**七次 panel 与 Probe 都是完全相同的四题集合**，且同一成员/同一 prompt/同一题的 resolved output 复用同一 request/cache identity。因此 3/3 的 target 收益转换不是独立复现，也不是新样本迁移；Probe 增加的是固定团队指标检查，未扩大 target preservation 的样本覆盖。11 次 Full 丢失同样全在 Probe 之外。

字段名称也需要谨慎解读：当前 `broad_delta` 在该路径直接等于 `target_delta`，不代表更广的独立题集；序列化的 `oracle_delta=0` 是未绑定的诊断默认值，未进入 promotion，不能当成实际 Oracle 无增益。由已有固定 peers 与 target 正确性重建，三个 positive Probe 的实际 Oracle 都是 1→2、delta +1，Vote 仍为 1。见 [字段解释与逐候选重建](probe_metric_interpretation.json)。未修改这些运行字段。

这里 additive repair 的定义是 **Full 保留全部旧正确集合且新增正确题**，不是文本 append-only。历史手工 P2 在独立的两次 Optimize60 realization 中均为保留 22、新增 3、总计 25/60；其 517 字符 procedure 限定代数/方程/函数触发，保留符号简化、精确运算、可行分支和其他题型的直接求解退路。两个 A4 Full 没有复现这个覆盖属性。P2 不是本次冻结 optimizer 的输出，也不是相同 request realization 的配对因果对照。不能用这次零 API 审计给未测候选赋分，或宣称 optimizer 的容量上限。

## 后续优先级（建议，未实施）

当前 Layer1 已有 correct anchor、broken-count 记录和 preservation 排序；local incumbent 也按 panel 正确数选择。因此本次不是证明“完全没有 preservation 机制”，而是观察到现有证据覆盖没有识别跨题损失。现有数据不能因果区分改采样、改目标权重与改 mutation 约束的效果。

1. 优先研究 repair gain 与 preservation risk 的局部联合目标及保留证据覆盖；当前一个 anchor 无法识别这两次 Full 的跨题丢失。
2. 同时研究 mutation 的适用域、通用求解退路与 optimizer-context 引用控制，避免生成无法由实际 Solver 输入执行的诊断流程。
3. export 没有实际严格净收益错排证据；promotion 同分限额保留为未测问题，不能优先宣称 selection bug。
4. Gradient/Pattern 的语义区分力与 Solver 稳定性仍需独立设计验证；现有短答案无法识别内部根因。

以上若进入实现，均需要明确新方法版本、冻结新身份及对照实验。当前 V2.2 保持原样；本报告没有更换 optimizer、提高 generation 数或续跑 Pilot。

## Full 丢失集中在哪里

11 次丢失去重后只有 **6 道**；G6 的 5 道丢失全部包含在 G4 的 6 道中，两者新增的 2 道也完全相同。按题型分母比较如下；样本极小，不作显著性或泛化推断。

| 来源题型 | 初始正确题数 | G4 丢失 | G6 丢失 |
| --- | --- | --- | --- |
| Algebra | 6 | 2 | 1 |
| Intermediate Algebra | 4 | 2 | 2 |
| Number Theory | 2 | 1 | 1 |
| Prealgebra | 7 | 1 | 1 |
| Counting & Probability / Geometry / Precalculus | 各 1 | 0 | 0 |

7/11 次发生在 Algebra 与 Intermediate Algebra。人工阅读原题与参考解后，六类任务/可见症状如下；“参考解所需操作”不代表已观察到 Solver 在该内部步骤犯错。

| 脱敏任务类别 | 丢失次数 | 可见症状 | 参考解所需操作 |
| --- | --- | --- | --- |
| 二次方程系数与根关系 | 2 | 最终值符号相反 | 系数符号与公式 |
| 嵌套根式化简 | 1 | 错误数值 | 精确符号化简 |
| 有理化与望远镜求和 | 2 | 一次错误数值、一次 terminal-invalid 表达式 | 符号变换、边界项 |
| 离散取值的极值 | 2 | 非最优值 | 因式分解、可行分支、极值检查 |
| 非十进制带符号减法 | 2 | 符号与大小均不正确 | 进位制运算与符号 |
| 分数应用题 | 2 | 数值内容吻合，但含叙述的 payload 非法 | 不可变输出契约 |

11 次中 **8 次 valid-wrong、3 次 terminal-invalid**。其中两次分数题并非可见数值知识丢失；它们包含正确数值，却输出了非法叙述 payload。剩余一次 terminal-invalid 的可见表达式数值也不正确。三次 invalid 均耗尽四次语义尝试，且各自四次内容完全相同。按冻结 parser，这三次仍为 incorrect；本审计没有 fallback 提取、改 parser 或重标正式正确数。

这些损失指向符号/结构处理、分支保留、算术符号与输出符合性，未形成“几何题集中退化”的证据。两个 child 共享的 5 道丢失中，有 3 道终值内容完全相同；其余两道丢失的内容不同。重复丢失集合削弱了把全部现象泛称为随机波动的解释，但仍不能识别 prompt 规则的因果贡献。

## 已有稳定性证据的边界

初始相同 prompt 有 5 个不同 member realization lane、共 300 个已实现的 Solver 逻辑请求身份。60 题中，4 题 raw content 不同、3 题 parsed output/有效性不同，但 **正确性分歧为 0/60**，五条正确集合完全相同。这个事实支持初始正确覆盖在已有 realization 中稳定，同时允许错误输出有变化。

两个 child 各只有一个完整 Full realization；Local/Probe/Full 的相同请求 cache 复用不能当成独立 replicate。terminal-invalid 的四次恢复尝试也不是完整 Full 重复。因此现有数据既不能证明随机波动导致能力损失，也不能给 child 评估方差、统计显著性或 latent reasoning 因果结论。稳定性归因为 **INCONCLUSIVE**。

33 条合格候选中，27 条改变了 local raw/parsed-or-validity 输出，只有 8 条改变正确性（3 正、5 负），另有 6 条 prompt 改变但 panel 输出不变。文本改写、行为改写与正确性改善是三个不同指标。

完整 [42 行候选表](candidate_table.md) 与 [candidate_metrics.json](candidate_metrics.json) 可逐行核对；七轮排名见 [local_ranking.json](local_ranking.json)，人工语义标记见 [manual_semantic_review.json](manual_semantic_review.json)，Full 覆盖见 [full_preservation.json](full_preservation.json)，11 次丢失归类见 [loss_case_review.json](loss_case_review.json)。其余完整审计、稳定性边界与验证记录随本报告保存。

## 工程验证与证据封存

标准命令显式限定 `tests --suite current`，在先于 application import 的离线网络 guard 下完成：**1,605 passed、2 skipped、1,468 deselected**，无 failure/error，网络尝试为 0。明确排除的历史/私有用例未执行，不宣称全部历史 replay 通过。生产模块及 scripts 的 compileall、当前 repository governance/manifest 索引、七轮 export 和 production promotion 重放、原始缓存评分核对、脱敏、账本及原始证据字节保留、`git diff --check` 均通过。完整分类、命令及日志/JUnit 哈希见 [verification.json](verification.json)。

验证过程中两次调用配置有误，已保留而未计作通过：首次未限定 tests 根目录，误收集忽略的运行副本，产生 153 个 collection errors；第二次全局 evidence capture 路径与历史夹具的临时 workspace 路径约束冲突，观察到 24 个断言失败后中止。去掉该 capture 参数，24 个相关单例全部通过；负对照复现了预期路径断言失败；最终标准 current suite 全部通过。未改测试或生产代码来消除这些失败。

旧治理 CLI 在既有 failure registry 兼容性校验处报 3 个错误并中止（缺少 symptom 一项、evidence_level 不兼容两项），其文件与父发布提交相同，未报告为通过。当前治理审计通过。原冻结 source commit 与科学/runtime 输入通过核对；已关闭尝试的当前治理 metadata 因停止发布和本审计而变化，不是可恢复执行的 startup identity，也未重新开放授权。

发布仅包括本报告、回溯协议、registry/lineage/frontier 和派生索引。原报告及两个既有尝试的原始证据保持字节不变。逐文件 SHA-256 见 [sha256_manifest.json](sha256_manifest.json)。
