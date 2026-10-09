# MATH Baseline Calibration Audit

零 API 审计完成；A/B/C 仅设计冻结。任务目录日期沿用 20261009，实际完成日期为 2026-10-10。
审计基线源码 `9bbe1b502a3c8c6d43b2da959b8b8b0f2256b985`；旧 Canary 执行源码 `7217081d43fff2e94ab46ed28378717e3e912520`。
诊断工具由 [provenance.json](provenance.json) 的逐文件 SHA256 冻结，所在 Git 提交提供完整复现源码。
本报告是证据，不改变 CURRENT_SPEC 或原实验。

## 核心结论与证据等级

**VERIFIED**：43 条最终无效回答中，37 条存在不看 Gold 即可提取、解析的明确数学结果；
其中 28 条正确、9 条错误。4 条候选不可解析，2 条没有足够证据。
这 28 条占最终无效逻辑画像的 65.12%。这是诊断 recoverability，不是新 baseline。
按每次生成先独立提取再评分，43 条最终无效画像中有 37 条至少一次非截断生成包含正确明确结果；
该存在性统计不是 Gold 选择的最佳答案评估，不能用于 Primary、Vote 或未来选样。

**OBSERVED / VERIFIED**：官方五成员正确数仍为 3、3、4、4、3 /12；Vote=Oracle=5/12。
60 条是五成员×12题的逻辑画像；192 条是成功语义响应，不是 192 道题。
17 条最终有效且正确、43 条最终无效、35 条重复相同 native reason、2 次超时，全部与旧报告一致。
35 不表示另有35道题，也不等于43条失败都保持同一种原因。

**INFERRED**：当前低官方分数明显受到输出接口损失影响。无法据此量化模型的总体数学能力，
也不能断言更明确 Prompt 一定解决问题。主要错误是标签与结果不在同一终态行，
需要独立 A/B Prompt-only 配对验证；C 是完整输出接口对照。
没有证据支持在本任务中改 Layer2、打开 Thinking、修改生产 Parser 或重算旧分数。

**UNKNOWN**：真实后端权重 revision、服务器内部是否忠实执行 thinking 控制、
新60题上的 A/B/C 有效性和货币价格。不能把请求别名等同权重版本。

## 格式分布与分母

首次60条生成：有效15，标签不在最后一行34，截断4，payload不合规5，Markdown包装1，boxed-only1。
最终43条无效画像的互斥主因如下；其他建议类别实际计数为零。

| 主因 | 数量 /43 | 比例 |
|---|---:|---:|
| FINAL_LABEL_NOT_LAST | 36 | 83.72% |
| MARKDOWN_WRAPPED_LABEL | 1 | 2.33% |
| MALFORMED_FINAL_PAYLOAD | 4 | 9.30% |
| OUTPUT_TRUNCATED | 2 | 4.65% |

192条成功响应的 native reasons：缺少合规显式终态141、payload歧义19、截断13、数学解析失败2、有效17。
两次数学解析失败发生在中间生成，不属于最终43条的独立数学解析失败类别。
诊断主因尊重原生 reason 与 finish reason：截断优先，payload问题优先于标签细分。
缺少合规终态进一步分为标签非末行135、包装3、boxed-only3。两套原因不能混算。
第一轮与终态、attempt与logical、native与diagnostic完整分布见
[format_failure_summary.json](format_failure_summary.json)。未发现缺失原文，INSUFFICIENT_EVIDENCE=0；
2条截断回答无明确可诊断结果，不能推测其后续答案。

## 恢复的边际收益与成本

| 语义轮次 | 条件 eligible /原始60 | 新增有效且正确 | 条件有效率 | 实际 reported tokens | 超时保守 charge |
|---|---:|---:|---:|---:|---:|
| 1 | 60/60 | 15 | 25.00% | 53,092 | 0 |
| 2 | 45/60 | 1 | 2.22% | 58,502 | 21,678 |
| 3 | 44/60 | 1 | 2.27% | 51,886 | 0 |
| 4 | 43/60 | 0 | 0.00% | 46,380 | 0 |

新增132次格式生成消耗156,768 tokens，只新增2条官方正确画像，即78,384 reported tokens/新增正确。
两次超时另计21,678 tokens保守charge；含超时的额外成本178,446，即89,223/新增正确。
这是整个恢复流程的平均边际成本，不证明某次超时必然产生这些实际tokens。
35条重复 native reason 画像消耗112,528 reported tokens，其中额外恢复84,855。
13次截断生成消耗63,883；其后11次容量扩展生成消耗59,469。这些计数重叠，不能相加。
终态2次截断没有后续恢复。容量扩展只由上一次成功生成截断触发，不改变 Thinking 或采样。
各主因token成本见JSON。全初始成功响应209,860，含两次超时保守charge共231,538。
旧完整Canary封账1,171,351与初始画像成本分属不同范围；本任务新API与新charge均为0。

四次规则确有微小恢复收益，但这批数据的后续条件有效率约2%、2%、0%，重复费用占主导。
不改变历史停止/重试策略；未来独立校准同时报告 First-draw 和 Recovered，以测量真实差异。

## 部署与 content 完整性

全部1,043个 Solver 物理请求和1,039个成功回执已核对：固定采样、thinking=false均PASS；
响应模型字段全部为qwen3-8b。HTTP发送路径直接发送 `serialized_request` 的同一UTF-8字节，
`extra_body`字段展平到HTTP JSON顶层，未回落到默认值。
1,039份普通Assistant content、finish reason、usage与Broker保存值一致；未出现 reasoning_content 字段。
初始60条轨迹与192条缓存/回执/trace/Parser输入逐字一致，未发现中间截断、丢内容或转义改写。
原生Parser重放与持久化预测一致。
矩阵及验证范围见 [solver_deployment_audit.json](solver_deployment_audit.json)。
thinking=false发送已VERIFIED；服务器内部执行与权重revision仍UNKNOWN。
Provider profile与别名映射由冻结binding及现有回执核对，不公开endpoint或host信息。
Immutable receipt先保存，trace随后增加usage可靠性归一化字段；科学内容与请求相等，
无需也未改动旧回执来强求运维metadata逐字段相等。

## 独立配对校准与预算

[完整协议](baseline_calibration_protocol.md) 和
[预算](baseline_calibration_budget.json) 已冻结；机器身份在
`experiments/protocols/math_baseline_calibration_v1/protocol.json`。
Optimize150排除Canary12后合法池138，确定性subject/level分层选60，所有reference通过现行解析检查。
哈希成员身份与实际分布记录在协议JSON，原始题目、Gold、逐题结果与响应留在ignored私有目录。
全臂thinking=false、同一model/采样/User问题/题目顺序/数学评分器。
A/B共用现行Parser，仅Answer指令不同；C使用独立严格终态boxed Parser，A/C、B/C同时改变接口。
First-draw作为主要对照；最多4次invalid-only恢复、每draw最多21次transport、截断后6144容量完全相同。
格式合法但数学错误时停止。Common Diagnostic仅secondary，不参与native恢复或Team Vote。

Stage1单成员；Stage2五个独立realization、等权plurality、Member/Vote/Oracle，需另行review和授权。
Stage2不复制输出、不复用Stage1缓存。逐题比较改进与退步，按题bootstrap seed81/10,000次；
60题开发诊断不支持普遍显著性结论，也不构造优化资格分数门槛。

| 阶段 | 逻辑请求 | 成功语义上限 | 含传输的物理上限 | 历史rate场景charged tokens | 独立拟议cap |
|---|---:|---:|---:|---:|---:|
| stage1 | 180 | 720 | 15,120 | 694,614 | 1,450,000 |
| stage2 | 900 | 3600 | 75,600 | 3,473,070 | 7,220,000 |

按旧初始画像频率估计的物理API场景为 Stage1 582次（576成功、6超时）、
Stage2 2,910次（2,880成功、30超时）；首次正常语义生成分别180/900次。
这些也是成本场景，不是新协议调用量保证，真实新API目前为0。

Stage1/2无传输失败的最坏reservation envelope分别
7,378,836 / 36,894,180；
所有物理尝试均保守计费的极端envelope分别
154,955,556 / 774,777,780。
历史rate仅成本场景，不预测新Prompt或新题。input按HTTP UTF-8字节+4096的现行保守reservation估计，
不是实际tokenizer数。拟议cap为历史charged场景的2倍，加B/C相对A输入增量，再向上取整至10k；
不保证完成最坏四次流程，可能预算中止。每次调度先检验charged+reserved，预算不足fail closed，
只发布未完成事实，不比较不完整arm的效力。两阶段不得借用旧2Mscope，均无真实API授权。

## 决策与未解决问题

先审阅此诊断和Stage1独立预算，再单独授权确切来源/协议/成员/roles/provider/cap/新scope。
本次仅有fake和dry-run工具；尚无可执行的paid runner或READY handoff。
优先用A/B分离Prompt遵循效果；用C比较输出接口；保留common诊断用于区分format-only、
math correctness、mixed与inconclusive，并保留成对退步与成本证据。
若合法输出稳定但数学错误仍多，审查subject/level、评分语义与单Solver任务结构，
再另行设计Strategy/Role研究。若简单契约仍不稳定，先定位部署/接口一致性，保持thinking=false。
这些是后续决策路径，没有自动改生产Seed/Parser、自动重跑或人为最低准确率门槛。
目前没有A/B/C真实结果，不能声称B或C更优；不能用本诊断构造新的Team Vote或论文baseline。

## 复现、治理与验证

使用项目已有依赖完整的Python环境，在无credential、网络guard预先加载的子进程执行：

```powershell
python tests/formal_zero_api_runner.py --offline-command scripts/audit_math_baseline_calibration.py
python tests/formal_zero_api_runner.py --offline-command scripts/audit_math_baseline_calibration.py --dry-run
python tests/formal_zero_api_runner.py --offline-command scripts/audit_math_baseline_calibration.py --fake-synthetic
python tests/formal_zero_api_runner.py tests --suite current
```

完整审计需要原始私有evidence与合法Optimize数据；缺失时fail closed，不补采样。
公开报告只含计数/类别/metrics/hashes及用户明确指定的planned协议指令。
原Canary、恢复与旧报告、manifest/binding、现有生产模块逐文件hash保持不变；
注册表仅追加两个passive节点与对应lineage，原条目不变，current_frontier不变。
最终验证计数与检查证据见 [verification.json](verification.json)。
220个历史/私有artifact模块按current suite分类排除，未声称历史replay全通过。
发布范围为本地commit；此次最新任务明确不push。
