# 历史候选 Guard 误拒审计（2026-10-15）

**结论：已证实最新 C 存在误拒，影响其全部四次重试耗尽机会；不能据此推断被拒候选会提升 Validation。** 本次只读、离线，未修复 Guard，未改代码、配置、授权或任何历史产物，未调用模型、未重测 Solver/Validation，也未操作实验进程。

## 1. 范围、版本与证据完整性

主表使用同一聊天最近比较的 A/B Seed42 三重复训练和修正后的 C Seed42 三重复训练：各60机会。A/B 冻结源码 `241f725797080b1b5fc714834b222b675a09d0e8`；C 为 `211a05c569f676f40067ed67eb445b0e5a93d8a0`。**A/B 未运行对应 Test；C 的训练与最终 Test 已完成。它们的候选检查与证据接口并非完全相同，不能视为严格匹配的消融。**

同时追溯旧 Seed88/89 链、最大 Pattern 修正链和旧 Seed42 中止前缀。10个运行目录涉及9个冻结源码版本；807份生成日志按响应回执完整性哈希去重后为493次尝试，均用其历史生成器/Guard 原函数体及原始响应、请求上下文复现，**判定与重建请求均零差异、日志零缺配**。另外12份最早 V1 响应也离线复现，其中10份可与原/继承 checkpoint 判定核对，2份第9轮响应缺终端判定记录；单列而不补成历史通过/拒绝。Canary 无候选生成不是缺日志。

每份实际回执校验完整性封印，Guard Git blob 按冻结 scope 的 LF/CRLF 哈希核对，manifest 核对其历史基础契约及修订，未把当前实现当历史。最早链使用的 `executability_checks` 函数摘要前缀为 `a3dc3b7a30ab`，Seed42 框架为 `cbf136bc79bb`；V6/V7额外调用私有解答/响应/诊断检查。完整源码、manifest、原始响应位置与哈希见 [源码版本](guard_provenance.json)、[函数摘要](source_functions.json)、[manifest](manifest_evidence.json)、[逐候选索引](evidence_index.json)。可用 `git show <source>:<path>` 查看对应实现。

实际状态以 [运行清单](run_inventory.json) 和产物为准：旧框架 A/B 六条完成，旧 C 重复1完成17轮后被用户中止，重复2/3未启动；其续跑目录残留 RUNNING 但 `user_stop.json` 明确记录停止，本次进程查询未发现匹配的实验 Python 运行器。最新 C 三条20轮均完整。repeat5、full_diagnosis_v5、module_alignment_v6 仅有准备产物，无执行 scope/训练日志，不计为运行或零拒绝实验。最早 V1 Canary 完成，两次正式尝试中止。旧 gated 两种子六条及最大 Pattern 修正两条完成；继承前缀只计一次。Test 续跑不生成优化候选，不另入候选分母。

## 2. A/B/C 主对照

**单位为历史候选生成尝试/响应 draw，不是不同 Prompt 文本数；包括 JSON 错误、超长和未改动。** 通过/拒绝指进入测评前的生成合法性，**不是 Validation 接受/拒绝**。误拒率＝确定误拒尝试÷全部生成尝试；无法确认不计入确定误拒；各组机会分母均60。额外重试＝第2至4次生成尝试，初次不计。分组耗尽不是缺少日志推算，而与 checkpoint 的实际决策核对。

| 组 | 尝试总数 | 通过 | 拒绝 / 拒绝率 | 确定误拒 / 误拒率 | 无法确认（拒绝中） | 额外重试 | 耗尽 / 含确定误拒的耗尽 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 65 | 60 | 5 / 7.69% | 0 / 0.00% | 0 | 5 | 0 / 0 |
| B | 87 | 58 | 29 / 33.33% | 0 / 0.00% | 1 | 27 | 2 / 0 |
| C | 77 | 56 | 21 / 27.27% | 10 / 12.99% | 4 | 17 | 4 / 4 |

C 确定误拒占其拒绝的 **10/21＝47.62%**；另4/21无法确认。A/B未证实误拒不等于证明其绝无误拒。主表四类计数：确定协议违规40、确定误拒10、无法确认5、合法通过174，共229次。这里“确定协议违规”包括 JSON 不合法、超长和无修改，**不等同于泄题或作弊**；“合法通过”指原判通过且本次未建立明确违规，非数学策略正确性或语义安全完备证明。

| 原始拒绝原因（可多标签） | A | B | C |
|---|---:|---:|---:|
| parent_no_op | 5 | 12 | 3 |
| REFLECTION_OUTPUT_CONTRACT_INVALID | 0 | 15 | 3 |
| SYSTEM_PROMPT_OVER_LENGTH | 0 | 1 | 1 |
| example_copying | 0 | 1 | 0 |
| private_solution_or_response_copy | 0 | 0 | 14 |
| example_answer_lookup | 0 | 0 | 0 |
| optimizer_only_dependency | 0 | 0 | 0 |

各原因计数不作为独立分母；本批主表每个拒绝只有一个记录原因，故总和恰好等于拒绝数。完整逐重复统计见 [summary.json](summary.json)。

## 3. 典型误拒与人工语义裁定

`executability_checks` 会把候选全文与私有文本按空白分词、大小写折叠，任意连续8词相同即记 `private_solution_or_response_copy`。V7把 `observed_failure` / `diagnosis` 纳入私有文本，但这些字段可能引用**本来就在父 Prompt 中的输出格式**，或描述通用数学操作。Guard 不区分来源或新增内容。

已人工核查相关候选的完整改动与证据，而非仅凭命中词裁定：10次仅包含通用格式要求或可用于新问题的向量逐点替代方法，没有题目到固定答案的映射，也不需要推理时读取诊断。14次私有文本拒绝中，11次的触发片段已经存在于父 Prompt，说明保留原有内容也能因当轮诊断不同被拒；这是上下文敏感的误拒风险，不意味着父 Prompt 任意内容都天然合法。

| 耗尽轮次 | 命中性质（释义，不公开原文） | 确定误拒 / 无法确认 / 确定协议违规 | 可追溯示例 |
|---|---|---:|---|
| C_seed42_repeat01 第13轮 | 通用输出格式要求与诊断中引用的父 Prompt 重合 | 3 / 0 / 1 | [生成日志](../../runs/ablations/pattern_c_only_seed42_repeat3_v7/train_run/C_seed42_repeat01/round_13_generation_1.json)、[原始响应回执](../../runs/ablations/pattern_c_only_seed42_repeat3_v7/train_run/provider_receipts/9353.json)；案例 `a2c870381bd2` |
| C_seed42_repeat02 第18轮 | 通用输出格式要求与诊断中引用的父 Prompt 重合 | 2 / 2 / 0 | [生成日志](../../runs/ablations/pattern_c_only_seed42_repeat3_v7/train_run/C_seed42_repeat02/round_18_generation_2.json)、[原始响应回执](../../runs/ablations/pattern_c_only_seed42_repeat3_v7/train_run/provider_receipts/26583.json)；案例 `427ad27ed74e` |
| C_seed42_repeat03 第8轮 | 通用输出格式要求与诊断中引用的父 Prompt 重合 | 2 / 2 / 0 | [生成日志](../../runs/ablations/pattern_c_only_seed42_repeat3_v7/train_run/C_seed42_repeat03/round_8_generation_2.json)、[原始响应回执](../../runs/ablations/pattern_c_only_seed42_repeat3_v7/train_run/provider_receipts/33705.json)；案例 `032bd9c55771` |
| C_seed42_repeat03 第18轮 | 通用几何向量逐点递推方法与诊断描述重合 | 3 / 0 / 1 | [生成日志](../../runs/ablations/pattern_c_only_seed42_repeat3_v7/train_run/C_seed42_repeat03/round_18_generation_1.json)、[原始响应回执](../../runs/ablations/pattern_c_only_seed42_repeat3_v7/train_run/provider_receipts/40225.json)；案例 `85bf2c194f0c` |

另4次候选的触发片段同样是格式要求，但候选全文还把**当轮提供的真实参考答案值**写成格式示例；不能因触发片段通用就断言整条候选合法，故保守归“无法确认”。旧/新链的3次 `example_copying` 命中参数化方程组或符号根式，与具体题干有重合，同时也可能是可复用的数学模板；同样归“无法确认”，不由词面重合直接判违规或误拒。详见 [无法确认清单](undetermined.json)。

重点规则复核：493次有判定日志的历史尝试中，`example_answer_lookup` 和 `optimizer_only_dependency` **均无实际触发**。其历史正则分别能笼统命中“如果题目是某类问题”的条件句，以及含使用动词的数学梯度表达；这是代码层面的潜在误拒面，不能计成已经发生的误拒。本次未新增候选或以合成用例充当历史样本，也不以出现关键词认定违规。

## 4. 重试与优化机会损失

最新 C 的4次耗尽占60机会的 **6.67%**，每次至少有一个确定合法候选被误拒，因而**确证有4次机会没有让已有合法候选进入 Optimize/Validation 测评**。它们对应重复1第13轮、重复2第18轮、重复3第8和18轮；[checkpoint 与完整 draw 分类](affected_opportunities.json) 均显示 `candidate_evaluated=false`。这里的损失是测评机会，不是证明损失了4次可接受更新。

这些4轮共16次生成：10次确定误拒、4次无法确认、2次 JSON 协议错误。C 全部17次额外重试中，**7次紧接确定误拒发生，4次紧接无法确认的私有文本拒绝发生**；它们是历史相邻重试的归因计数，不是修复后可保证节省的调用数。A耗尽0；B耗尽2均来自已复现的 JSON/未改动失败，B那次无法确认的例文重合随后重试成功进入测评，未导致耗尽。无额外“仅由无法确认影响”的耗尽轮次。

误拒确实集中在当前 C；但 C 独有诊断字段、A/B使用较早且不完整的私有文本检查，组间也有生成内容和重试差异。**不能把这一差异直接解释为 Gradient–Pattern 天生更容易生成违规文本，或把修复后的潜在收益作为已观测机制效果。** 也不能离线推断被拒候选本来会提高 Validation，或据此改写任何最终分数。

## 5. 旧实验补充与缺失证据

下表只作历史覆盖说明，不与主表混合估计机制效果；各续跑前缀按真实响应去重，最大 Pattern 行只计其新增后缀。旧 C 不加入最新 C 成绩。

| 历史队列 | 机会 | 尝试 | 通过 | 拒绝 | 确定误拒 | 无法确认 | 额外重试 | 耗尽 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 旧 A（Seed88/89） | 40 | 86 | 29 | 57 | 0 | 0 | 46 | 11 |
| 旧 B（Seed88/89） | 40 | 49 | 40 | 9 | 0 | 0 | 9 | 0 |
| 旧 C：全部 Pattern | 40 | 64 | 35 | 29 | 0 | 2 | 24 | 5 |
| 旧 C：最大 Pattern 新增后缀 | 37 | 45 | 37 | 8 | 0 | 0 | 8 | 0 |
| 中止的旧 Seed42 C 前缀 | 17 | 20 | 17 | 3 | 0 | 0 | 3 | 0 |

以上合计493次，与主表互不重复。最早 V1 的12份响应另列 [V1重放证据](original_v1_replay.json)：10份与 checkpoint 原判一致，2份缺原始终端判定，虽可重放得出超长结果，也不将重放结果冒充原始拒绝，更不把缺失当零次。V1允许未改动且只有两次生成，与后续四次生成、必须修改的协议不同，因此不并入主分母。

## 6. 结果解释与建议（未实施）

现有准确率仍是实际保留 Prompt 的真实测量结果，但最新 C 的四个耗尽机会不能再统称为“优化器没能给出合法修改”。此前看到 C 更偏 Strategy、较少修改 Answer，也可能部分受到格式误拒影响；本审计没有量化这种方向偏差，更未证明其因果大小。A/B/C既有接口版本不一致和本次观察到的差别，均限制严格机制优劣解释。

建议优先修复**已证实**的父 Prompt 格式/通用诊断八词重合误拒，并为通用条件句、数学梯度与真正的题目答案查表/推理时私有证据依赖建立区别测试；不建议简单删除泄漏保护。若目标仍是比较机制效果，应在另行审阅、冻结与授权后使用统一 Guard 和证据接口重做匹配 A/B/C；不能补造或追认历史候选成绩。本次不修复、不续跑、不补测。

## 7. 验证与复查方式

验证范围为报告/审计证据：历史原函数离线重放、完整性封印、版本/manifest、分母/算术、链接、输出脱敏与文件哈希；不运行无关生产测试或全历史治理扫描。网络入口在审计进程中阻断，实际网络尝试0，真实API0，原始输入检查前后哈希一致。详情见 [verification.json](verification.json) 与 [artifact_hashes.json](artifact_hashes.json)。

公开报告只保留类别、计数、相对日志位置与哈希，不公开 Prompt、题目、答案、原始响应或端点。原文可在逐候选索引所指的既有私有 `provider_receipts` 中复查；`response_sha256` 为原响应 text 的 UTF-8 SHA256，`candidate_sha256` 为合法构造的三块候选对象的规范 JSON SHA256，`id` 为完整回执封印，不能混作同一种哈希。
