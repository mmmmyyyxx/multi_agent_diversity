# V2.8：Scoped Edit Experience 与 Layer1 Evidence Feedback

本次实现并离线验证 Memory 送达与 Layer1 证据反馈两项改进，建立新方法身份 `unified_team_prompt_search_v2_8_scoped_experience_v1`。真实模型利用、候选质量、单位 Token 效率和团队收益均未测量。没有调用模型 API，没有续跑历史实验，没有修改历史成绩，也没有 Push。

当前准备 manifest 为 DRAFT：source、数据/划分、执行 binding、授权与冻结闭包尚未设置，`READY_TO_RUN=false`。方法身份和实现登记不等于真实实验获准执行。

## 1. V2.7 为什么没有充分利用现有机制

依据 [Evolution Bottleneck Audit](../math_v27_seed86_evolution_bottleneck_audit_20261012/README.md)、封存的逐机会/候选/实际输入和执行源码，而非把 Commit 少预先归因于门禁。

| 环节 | 直接观察及代码机制 | 本次判断 |
|---|---|---|
| Recording | 原记录保留可重建编辑链、局部与 Full 结果；不是只有日志文本 | 首要问题并非没有经验。补充机会父团队、Mutation 类型和实际送达来源 |
| Retrieval/Selection | 跨机会选中 Pattern 的精确 ID 均不同；旧优先级让当前局部记录和负向 Full 占优 | 精确匹配不足以表达相关性；不能把“失败”作为普遍优先标签 |
| Delivery | 300 次 Reflection 中，240 次 bank 有 Full，只有 113 次 wire 含 Full；127 次遗漏中 34 次来自排序、93 次来自预算 | 已确认的直接信息流缺口，应先修复 |
| Content | 旧动作标签可能包含继承的整个 Prompt；假设/动作文本被截断，Full 视图过大 | 需要实际改动、测量范围、父状态适用关系和完整可读文本 |
| Layer1 | Reflection 收到广义 Pattern，未收到已有的细致 Mutation 诊断和 worked solution；16 次全弃权机会的六个请求相同 | 存在可修的信息衰减与反馈缺口，不等于已经识别弃权的数学原因 |
| Utilization | 旧记录没有足以识别“模型使用了哪条经验”的证据 | 不能仅凭 bank、wire 或输出相似性声称有效学习 |

V2.7 的 300 个槽位中有 190 次 NO_SAFE_EDIT、34 次契约无效、76 个可评估编辑；只有 7 个局部 SearchValidation 正增益编辑。这支持优先研究生成质量。既有审计没有发现已观察到的局部正增益被导出排序剪掉；25 个 Full 拒绝中 23 个 Vote 回退，不能据此放宽 Admission。后 23 次机会没有新 Commit 的原因不能由“Memory 未送达”单独解释，Solver realization、诊断正确性、修复迁移与评价代表性仍相互混杂。

## 2. 实际改变了什么

规范：[V2.8 amendment](../../docs/design/SCOPED_EDIT_EXPERIENCE_LAYER1_V28.md)。主要实现位于 `search/scoped_edit_experience.py`、`rolling_risk_memory.py`、`current_layer1.py` 和 `bounded_layer1.py`。

**Memory：可适用的实际编辑经验。** 同成员检索综合 Pattern/假设、规范化词项、实际变动概念、Mutation 数学类型以及机会根父/子 Prompt 和团队身份。全部为启发式相关性，不是机制正确或因果证明。优先最相关的已测 Full，再考虑相反结果的 Full 和局部经验；没有负向经验自动优先。完全相同的父/子/团队及测量观测可去重，冲突结果不合并。

卡片先提供具名范围与分母、实际编辑块/操作、Full fixed/lost/invalid、目标和 Vote 变化及机会根状态匹配。Full-supported 但 Shadow 拒绝不会标成 Commit。末次实际变动交叉的完整句子/子句可作为 `last_step_instruction`；这是末次编辑说明，不是全部编辑链，更不是该句的独立 Full 效果。完整文本无法容纳时整项省略，不让模型执行数字切片。

总预算仍为 canonical ASCII JSON 1200 字符，private/shared top-k 仍各 3，private 每成员 24、Full-refuted 上限 5、shared 48 均保持。主 Full 的完整末次编辑优先使用预算，句子上限 600 字符包含在总预算内；额外卡片、概念、类型、整条假设及 Shared Risk 使用余量。没有简单扩大预算或新增保留硬约束。

**Layer1：诊断到生成的证据连续性。** 只对选中 Pattern 支持内的 Mutation 样本送达 accepted root diagnosis、observed failure、reusable correction、advisory block 和 expected effect，标明其基于机会原 Prompt。当前局部父的普通 Solver 响应仍完整直接可读。Optimize worked reference 单独标注来源、原长度与 4096 字符前缀截断；不是 Solver 写作，也不是隐藏推理。

随后最多三次已完成尝试提供弃权/契约失败类别、实际编辑块及 Mutation、根 SearchValidation、实际父 SearchValidation 的测量计数。没有被拒文本、候选档案、未来反馈或原始 SearchValidation/Probe 问题。精确 replacement-block 容量让 Reflection 能在现有 3000 字符总限内考虑简洁重写。生成说明鼓励明确、条件化、可检验的修复和 scoped experience 使用，同时保留合理弃权及中性/负向局部探索。

记录实际 delivered IDs、Memory 输入 hash 和投递审计；它们证明输入来源，不推断 utilization。新字段与原始诊断进入 optimizer dependency/copy guard，抽象修复假设仍可用于可迁移规则。

**没有改变的机制：** WHO/Responsibility、Gradient 的主要生成/抽象语义、采样三组 3/3/6、6 次 Reflection、42 次局部 metric、4 个导出、2 个 Full 晋级、每次安全评估编辑推进局部父、候选排名、Solver/供应商/评分/aggregation、Full/Shadow、安全与泄漏保护、正式停止规则。没有 Preservation Gradient/Panel/新硬约束，没有 Pareto Archive，没有额外模型调用。

## 3. 历史静态投影：送达和内容

逐次匹配 350 个 Memory read 中的 300 个 Reflection-time bank 与实际 wire hash；排除每机会起始的 50 个非 Reflection read。只读投影相同历史 bank，未导入当前 Memory runtime/checkpoint，未重采 Solver、生成 Reflection 或评价新候选。类型和根身份取自原机会的 Optimize 元数据。分母是输入事件，不能当作 300 个独立实验。

| 同一组 300 个 Reflection 输入 | V2.7 已记录 wire | V2.8 视图投影 |
|---|---:|---:|
| bank 有 Full 的输入 | 240 | 240 |
| 至少一个 Full 实际送达 | 113/240 | 240/240 |
| 有完整末次编辑文本的 Full 输入 | 旧视图没有该字段 | 240/240 |
| bank 有 Commit 时至少一条 Commit 送达 | 4/180 | 102/180 |
| 有完整末次编辑文本，含局部卡 | 不作等价比较 | 267/300 |
| 至少一条 Shared Risk 送达 | 22/300 | 15/300 |

新上下文最大 1199、中位数 1173.5 字符。423 张 private card 在 300 个输入中送达；219 个输入仍省略至少一张 top-k 候选卡，48 个输入同时包含 supported/committed 与 refuted Full。历史完全相同观测去重触发 **0**，因此不能把历史送达改善归功于去重。

这个取舍并非所有信息都变多：完整旧修复假设在投影中没有整条送达，类型字段在 135 个输入中送达，Shared Risk 少了 7 个输入。模型可读实际末次条件化编辑、当前 Pattern、具名结果及根状态关系，但不保证所有历史假设或所有编辑链步骤可见。完整子句只是表面完整性，不能保证数学自足、可迁移或因果正确。相关性启发式可能过宽或过窄，存储仍有淘汰上限。

## 4. Layer1 投影与 Mock：能验证到哪一层

原 900 份 Mutation 当前普通响应保留。新增可送达的 accepted selected-support diagnosis 为 786 份，覆盖全部 50 次机会；worked reference 为 900 份，此批没有超过 4096 字符。250 个后续输入可获得 preceding-attempt metadata。没有读取 Validation/Test 原始数据，也没有向 Reflection 添加原始 SearchValidation/Probe 文本。

16 次全 NO_SAFE_EDIT 机会的投影输入均不再全部相同；三条窗口饱和后仍可能相同，未实施强制改写、提前终止或增加槽位。没有任何新输出证明这些弃权会减少。

恢复证据有成本。字段投影的 serialized JSON 字符增量为 min 2998、中位数 6896.5、max 9779；该数字还不含新生成说明的增量，**不是 provider billed Token、缓存收益或延迟实测**。不以少给必要证据交换省 Token。下一步应检验改善能否抵消输入成本。

新增回归覆盖：Full 不被近期局部卡挤掉、正负 Full 并存、冲突观测不合并、只识别实际变动、Unicode/长规则完整省略、无关 Full 不普遍化、当前基准 Commit 相关性、成员隔离、旧 schema/prefix 拒绝、Mutation-only diagnostics/reference、完整响应、准确容量、局部负增继续探索、42 metric/6 槽不变、弃权反馈、来源可审计、optimizer evidence 不进入 Solver，以及规范变化必须改变科学源 hash。旧 typed rejection 测试更新为允许新增 preceding-attempt metadata，拒绝文本守卫保持。

一个有条件 Fake consumer 在第二次同成员搜索读到 Full-refuted 经验后改变行为，证明 delivery→consumer 的接线有效。它按测试代码条件反应，不能替代真实 Reflection 使用 Memory 的证据，更不是候选/团队收益。

| 证据层次 | 本次结果 |
|---|---|
| Memory 送达 | 静态投影改善；生产接线离线验证 |
| 更具行动信息 | 实际末次完整编辑与 measured scope 可见；不保证规则正确 |
| Reflection 实际利用 | 只验证 Mock 条件输入响应；真实模型未测 |
| Candidate 质量与效率 | 契约、信息边界、探索与预算回归通过；真实有效编辑率/重复率/成本未测 |
| Team Vote/互补性收益 | 未执行新实验，不提供新成绩 |

## 5. 版本、治理及验证范围

Binding/cache/composition V11、Memory/Layer1 V4、optimizer input V12；observed-response policy 更新其 optimizer 输入绑定，Solver 投影及评分语义不变。新 amendment 纳入科学源哈希闭包。新 schema 严格分支追加；原有专用分支保持，通用 fallback 仅新增 V2.8 排除项，旧版本字段和合法性约束不改。历史 parse 不授予当前执行资格。

新准备节点及父 lineage 已登记，frontier 为 `NO_AUTHORIZED_REAL_EXPERIMENT`，无当前 executable scope。旧 V2.7 continuation policy 在当前 contract 中明确拒绝，旧 Memory 记录、checkpoint、cache realization 和授权不复用。旧实验、成绩、manifest、三份相关报告与封存轨迹的不变性已核验。

完整 current suite 选择 810 个用例，首次结果 803 passed / 5 failed / 2 skipped。五项失败分别是旧 current-preparation 路径、fallback 对新版本的排除和三个旧 Memory 字段集合断言。修复后四个受影响模块的 95 项回归全部通过；运行时和规范在该完整检查后没有再修改。最终覆盖合计 **808 passed / 2 skipped**，这是一次完整运行加受影响回归的合并证据，不是声称重新运行了一次全绿套件。完整检查约 24 分 36 秒；未因发布而重复。

生产 compileall、current repository governance、当前 DRAFT manifest 与 invariant/failure/lineage/generated views 检查通过。复用未变的历史 manifest/report/trajectory 不变性检查；新 schema 排除项之后重新审计 current governance。旧治理 CLI 此次也通过，不沿用之前 40 个错误作为当前状态。明确排除 220 个历史/私有模块，未执行全历史 replay。

所有应用诊断/测试均在去凭据的网络守卫下执行，网络尝试 0，真实模型 API/Validation/Test 调用 0。sanitization、公开文件字节 hash 和 diff whitespace 检查结果见 [verification](verification.json)。历史/私有测试的排除不等于历史 replay PASS。

## 6. 最小后续研究及判别标准

第一步独立冻结小规模受控机制诊断，采用 **Memory view × Layer1 evidence/feedback 的 2×2**。例如预先选定 8 个 Optimize 机会根状态，覆盖有成功/拒绝经验、全弃权和容量受限的类别；各 cell 使用相同机会/角色证据和同样 6 槽/42 metric，不增加单次生成预算。各 cell 必须具有明确独立 treatment/source/input/cache 身份和新授权，不能靠 current runtime compatibility mode 或恢复旧实验执行。固定历史案例只能支持条件机制判断，不能代表泛化。

Memory 对照比较旧/新投递视图；Layer1 对照比较 generalized-only 与本次 rich diagnostic/reference/attempt/capacity 输入。完整有用证据不能故意以切片交付。比较 delivery、可行动内容、真实引用/采纳/拒绝的可审计依据、合同合法且非重复编辑比例、NO_SAFE_EDIT、局部修复/丢失/invalid、完整候选的 target/Vote/独有覆盖变化、输入/输出 Token 和每个有效候选成本。自然文本自称“用了 Memory”或表面片段重叠不足以证明利用，应结合输入干预、实际 diff 与效果；评估解释不得回流搜索。

2×2 先分离经验投递与 rich Layer1 的总体作用，不把四个 Layer1 子字段都归因为一个已识别机制。若 rich Layer1 有效但输入成本过高，再冻结同证据范围的分字段消融，决定是否有可安全省略的信息；不能先牺牲角色能力来优化成本。

第二步才是新授权的短程 sequential A4 对照，比较完整更新链和后期停止前的有效候选成本；固定父诊断不能证明持续团队演化。Full/Shadow、stop、WHO、Solver 和评分维持一致。按事前预算/停止运行，未达科学停止应报告资源截断。

| 后续方向 | 预期科研价值 | 现有证据 | 复杂度/Token 成本 | 优先级 |
|---|---|---|---|---|
| 验证 Memory 与 Layer1 两项 treatment | 高，直接检验确认的信息缺口 | 送达/接线充分；真实利用未测 | 小规模 factorial，中等调用成本 | 1 |
| 诊断错误/修复迁移及反馈利用 | 高，可能解释生成无效 | 语义质量和能力因果未识别 | 抽样审阅已有输出低成本；真实干预需另冻 | 2 |
| Candidate Selection 的 Full 预测信息 | 中高，避免遗漏未知优质候选 | 没有已观察正局部被剪掉；未 Full 候选未知 | 补 Full 诊断成本较高 | 条件性 |
| Preservation 新机制 | 潜在高，需先区分自然修复质量 | 存在回归，尚不能确认其为主瓶颈 | 新科学机制和对照，成本较高 | 暂缓 |
| 放宽 Team Admission | 当前价值低，风险明确 | 已拒候选大多 Vote 回退 | 不能通过增 Commit 判断收益 | 不建议 |

下一阶段优先验证并继续改善 Layer1 的真实经验利用。只有在送达已稳定、模型能产生可验证修复而 Full 仍系统退化时，才有更充分理由研究 Preservation 或选择证据；如果大量候选仍无有效修复，应继续诊断生成/诊断迁移。现有证据没有支持修改 Team Admission。

公开附件：[Memory view projection](memory_projection.json)、[Layer1 field projection](layer1_projection.json)、[classifier](classifier.json)、[verification](verification.json)、[byte hashes](sha256_manifest.json)。没有原题、答案、具体实验 Prompt、原始响应、供应商 endpoint 或主机路径。
