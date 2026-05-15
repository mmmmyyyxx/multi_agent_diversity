# P3 GPT-5.5 Normal-Judge Rejudge

目标：抽样 `target` 不包含 `option_contrast`、但原自动 judge 的 primary 标签为 `option_contrast` 的 trace，让 GPT-5.5 在接近正常 judge 的输入条件下重判。

GPT-5.5 输入包含：完整 taxonomy leaf labels、major-family tree、base family definitions、reasoning_summary 要求、confidence/evidence 规则、返回 JSON schema、agent_id、task_type、question_hash、trace_hash、trace_length、extracted_answer、Single agent trace。

GPT-5.5 输入不包含：目标策略、模型身份、run 名称、原自动 judge 标签、gold answer、group/vote 信息。

- normal_judge_context_file: `p3_normal_judge_context.json`
- candidate_count: 722
- sampled_count: 40
- evaluated_count: 40
- 每条 `p3_normal_judge_packet.jsonl` 也内嵌 `normal_judge_equivalent_context`，方便逐样本审计。

判读规则：

- 如果 GPT-5.5 也把 primary 判为 `option_contrast`：原 judge 的 option_contrast 判定被支持，更像模型/trace 本身确实是 option-style。
- 如果 GPT-5.5 判为非 `option_contrast`：原 judge/taxonomy 的 option_contrast 吸附问题被支持。
- 如果 GPT-5.5 把 `option_contrast` 放在 secondary：说明 trace 是混合策略，原 judge 可能把次要的选项格式提升成 primary。

## Overall

| n | GPT primary option | GPT pair option | original judge supported | judge/taxonomy questioned | confidence |
|---|---|---|---|---|---|
| 40 | 0.0750 | 0.1750 | 0.0750 | 0.9250 | 0.8960 |

## By Target

| agent | target | n | GPT primary option | GPT pair option | original judge supported | judge/taxonomy questioned | confidence |
|---|---|---|---|---|---|---|---|
| 0 | concept_definition_match | 10 | 0.0000 | 0.1000 | 0.0000 | 1.0000 | 0.8470 |
| 2 | answer_to_stem_backward_check\|option_contradiction_check | 10 | 0.2000 | 0.2000 | 0.2000 | 0.8000 | 0.9120 |
| 3 | rule_or_principle_application | 10 | 0.0000 | 0.2000 | 0.0000 | 1.0000 | 0.9120 |
| 4 | decomposition\|stem_evidence_alignment | 10 | 0.1000 | 0.2000 | 0.1000 | 0.9000 | 0.9130 |

## By Model And Target

| model | agent | target | n | GPT primary option | GPT pair option | judge/taxonomy questioned |
|---|---|---|---|---|---|---|
| deepseek-chat | 0 | concept_definition_match | 3 | 0.0000 | 0.0000 | 1.0000 |
| deepseek-chat | 2 | answer_to_stem_backward_check\|option_contradiction_check | 3 | 0.0000 | 0.0000 | 1.0000 |
| deepseek-chat | 3 | rule_or_principle_application | 3 | 0.0000 | 0.0000 | 1.0000 |
| deepseek-chat | 4 | decomposition\|stem_evidence_alignment | 3 | 0.0000 | 0.0000 | 1.0000 |
| gemini-2.5-flash-lite | 0 | concept_definition_match | 3 | 0.0000 | 0.3333 | 1.0000 |
| gemini-2.5-flash-lite | 2 | answer_to_stem_backward_check\|option_contradiction_check | 3 | 0.6667 | 0.6667 | 0.3333 |
| gemini-2.5-flash-lite | 3 | rule_or_principle_application | 3 | 0.0000 | 0.3333 | 1.0000 |
| gemini-2.5-flash-lite | 4 | decomposition\|stem_evidence_alignment | 3 | 0.3333 | 0.3333 | 0.6667 |
| gpt-4o-mini | 0 | concept_definition_match | 2 | 0.0000 | 0.0000 | 1.0000 |
| gpt-4o-mini | 2 | answer_to_stem_backward_check\|option_contradiction_check | 2 | 0.0000 | 0.0000 | 1.0000 |
| gpt-4o-mini | 3 | rule_or_principle_application | 2 | 0.0000 | 0.0000 | 1.0000 |
| gpt-4o-mini | 4 | decomposition\|stem_evidence_alignment | 2 | 0.0000 | 0.5000 | 1.0000 |
| qwen2.5-7b-instruct | 0 | concept_definition_match | 2 | 0.0000 | 0.0000 | 1.0000 |
| qwen2.5-7b-instruct | 2 | answer_to_stem_backward_check\|option_contradiction_check | 2 | 0.0000 | 0.0000 | 1.0000 |
| qwen2.5-7b-instruct | 3 | rule_or_principle_application | 2 | 0.0000 | 0.5000 | 1.0000 |
| qwen2.5-7b-instruct | 4 | decomposition\|stem_evidence_alignment | 2 | 0.0000 | 0.0000 | 1.0000 |

## GPT-5.5 作为 Taxonomy Judge 时的目标策略命中

上面的表只回答了“原 judge 的 `option_contrast` 是否被 GPT-5.5 支持”。但更关键的问题是：如果用 GPT-5.5 在同一 taxonomy 下重新贴标签，trace 是否命中了原始目标策略。这里把 target leaf 与 GPT-5.5 给出的 `primary_family` / `secondary_family` 做匹配。

指标含义：

- `primary exact target hit`：GPT-5.5 的 primary leaf 是否正好落在 target leaf 集合中。
- `pair exact target hit`：GPT-5.5 的 primary 或 secondary 任一 leaf 是否落在 target leaf 集合中。
- `primary same-major hit`：GPT-5.5 的 primary leaf 所属 major family 是否落在 target major 集合中。
- `pair same-major hit`：GPT-5.5 的 primary 或 secondary 任一 leaf 所属 major family 是否落在 target major 集合中。

### Overall Target Hit

| n | primary exact target hit | pair exact target hit | primary same-major hit | pair same-major hit |
|---|---|---|---|---|
| 40 | 0.1250 | 0.1250 | 0.6000 | 0.7000 |

### By Target

| agent | target | n | primary exact target hit | pair exact target hit | primary same-major hit | pair same-major hit |
|---|---|---|---|---|---|---|
| 0 | concept_definition_match | 10 | 0.5000 | 0.5000 | 0.6000 | 0.8000 |
| 2 | answer_to_stem_backward_check\|option_contradiction_check | 10 | 0.0000 | 0.0000 | 0.8000 | 0.9000 |
| 3 | rule_or_principle_application | 10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 4 | decomposition\|stem_evidence_alignment | 10 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |

### By Model

| model | n | primary exact target hit | pair exact target hit | primary same-major hit | pair same-major hit |
|---|---|---|---|---|---|
| deepseek-chat | 12 | 0.1667 | 0.1667 | 0.6667 | 0.7500 |
| gemini-2.5-flash-lite | 12 | 0.0000 | 0.0000 | 0.5833 | 0.8333 |
| gpt-4o-mini | 8 | 0.2500 | 0.2500 | 0.6250 | 0.6250 |
| qwen2.5-7b-instruct | 8 | 0.0000 | 0.0000 | 0.5000 | 0.5000 |

这组命中率说明：GPT-5.5 并不支持原 judge 的 `option_contrast` primary，但它也没有把大多数样本精确贴回目标 leaf。整体 exact target hit 只有 0.1250；不过 same-major hit 达到 0.6000/0.7000，说明 GPT-5.5 往往认为 trace 位于目标策略附近的大类，而不是完全无关的方法。

按 target 看，`concept_definition_match` 是唯一 exact hit 明显恢复的 leaf，pair exact 为 0.5000。`answer_to_stem_backward_check|option_contradiction_check` 的 exact hit 为 0，但 same-major 高达 0.9000，说明 GPT-5.5 认为这些 trace 仍在 MMLU option-semantics 大类内，只是不一定精确遵循“反向检查/矛盾检查”这两个 leaf。`decomposition|stem_evidence_alignment` 的 exact hit 也为 0，但 same-major 为 1.0000，说明很多 trace 被判到同一目标大类中的邻近策略。`rule_or_principle_application` 最弱，exact 和 same-major 都为 0，说明该目标要么 prompt 遵循不足，要么 taxonomy 下的 leaf 边界与实际 trace 表达严重错位。

## 结果解读

这组复判强烈支持“原自动 judge 把 `option_contrast` 过度提升为 primary 标签”的解释。样本设计中，40 条 trace 的原自动 judge primary 都是 `option_contrast`，但 GPT-5.5 在拿到完整 taxonomy、major-family tree、family definitions、confidence/evidence 规则和同等单 trace 信息后，只有 3 条仍把 primary 判为 `option_contrast`，比例为 0.0750。把 primary 或 secondary 任一位置含 `option_contrast` 都算上，也只有 7 条，比例为 0.1750。

换句话说，在这 40 条高风险样本里，原 judge 完全被 GPT-5.5 支持的只有 3/40；有 4/40 更像是“trace 确实有选项比较成分，但它只是 secondary”；剩下 33/40 在 GPT-5.5 看来 primary 和 secondary 都不该是 `option_contrast`。同时 GPT-5.5 的平均 confidence 为 0.8960，说明这不是低信心、边界模糊导致的随机差异。

按目标策略看，`concept_definition_match` 和 `rule_or_principle_application` 两组最能说明问题：这两组各 10 条中，GPT-5.5 的 primary `option_contrast` 比例都是 0。也就是说，原 judge 把它们标成 `option_contrast`，更可能是被 MMLU 多选题的“逐项看选项”表面格式吸引，而没有抓住 trace 的主导方法。`answer_to_stem_backward_check|option_contradiction_check` 组的 primary `option_contrast` 比例较高，为 0.2，因为这类策略本身更接近 option-semantics，和选项比较有天然重叠。`decomposition|stem_evidence_alignment` 组为 0.1，说明 decomposition/evidence alignment prompt 下也有少量 trace 真实地退化成了选项对比。

按模型看，问题不是所有模型都同等严重。deepseek-chat、gpt-4o-mini、qwen2.5-7b-instruct 的抽样样本中，GPT-5.5 判为 primary 或 pair `option_contrast` 的比例都是 0；只有 gemini-2.5-flash-lite 出现明显的真实 option-style trace，primary 比例 0.25，pair 比例 0.4167。因此，对大多数模型而言，原 judge 的 `option_contrast` primary 更像误判；对 gemini 而言，部分 trace 可能真的更偏选项比较。

因此，这个实验更支持“judge/taxonomy 使用方式有问题”，尤其是 cheap/normal judge 对 `option_contrast` 的吸附或主次排序问题；不支持“这些 trace 本来就都是真实 option_contrast”。但如果只看 GPT-5.5 taxonomy label 的 exact target hit，也不能说模型已经精确命中了目标 leaf，因为整体 exact target hit 只有 0.1250。更合理的说法是：原 judge 的 `option_contrast` primary 多数不成立；GPT-5.5 认为不少 trace 落在目标 major 附近；是否真正遵循原始策略指令，还需要独立的 prompt-following 评审来判断。

后续已经运行 `prove_experiments/p3_prompt_following_gpt55/p3_prompt_following_summary.md` 中的独立 prompt-following 评审。该评审让 GPT-5.5 只看原始策略指令和 trace，结果显示 `followed_rate=0.7250`，`partial_or_better=0.8750`，`judge_taxonomy_likely=0.7000`，进一步支持“多数高风险样本不是模型完全没遵循，而是自动 judge/taxonomy 把选项形式过度贴成了 `option_contrast`”。
