# 受控 Probe 的 Prompt 组

这些 prompt set 用于 P2 和 P3。正式运行前建议保存为 JSON，供最小 probe runner 读取。

## P2 同策略改写：Option Elimination

目标 family：

- `distractor_elimination`
- `option_contrast`

Agent prompts：

1. Use option-by-option elimination. Check each candidate answer against the question, discard inconsistent choices, and output exactly one `FINAL_ANSWER:` line.
2. Compare all answer choices with the stem. Reject distractors first, then choose the remaining best-supported option. End with exactly one `FINAL_ANSWER:` line.
3. Treat the options as hypotheses. Test each one for contradictions with the question, remove failures, and select the surviving answer. End with exactly one `FINAL_ANSWER:` line.
4. Before answering, list why each incorrect option fails. Choose the option least contradicted by the facts in the question. End with exactly one `FINAL_ANSWER:` line.
5. Solve by eliminating impossible choices rather than by free-form recall. Check every option and then output exactly one `FINAL_ANSWER:` line.

期望结果：

prompt 措辞多样性较高，但 strategy-tree diversity 较低。

## P3 混合策略：MMLU

Agent 0 target：

- `concept_definition_match`

Prompt：

Identify the central concept or definition tested by the question. Match the stem to the relevant definition or concept boundary before considering options. End with exactly one `FINAL_ANSWER:` line.

Agent 1 target：

- `distractor_elimination`
- `option_contrast`

Prompt：

Compare the answer choices one by one. Eliminate distractors that conflict with the stem, then choose the best remaining option. End with exactly one `FINAL_ANSWER:` line.

Agent 2 target：

- `answer_to_stem_backward_check`
- `option_contradiction_check`

Prompt：

Work backward from each plausible answer to the question stem. Ask whether the stem would still be true if that answer were selected, and reject options that create contradictions. End with exactly one `FINAL_ANSWER:` line.

Agent 3 target：

- `rule_or_principle_application`

Prompt：

Identify the governing rule, theorem, principle, or domain law. Apply that rule explicitly to the facts in the stem before choosing an option. End with exactly one `FINAL_ANSWER:` line.

Agent 4 target：

- `decomposition`
- `stem_evidence_alignment`

Prompt：

Break the stem into facts, constraints, and implications. Align each important fact with the options before selecting the answer. End with exactly one `FINAL_ANSWER:` line.

期望结果：

family diversity 应高于 P2，并且完整 trace 中能看到方法差异。

## P3 混合策略：数学/定量变体

适用于 GSM8K、AQuA-RAT 或 MMLU 定量子集。

目标 family：

- direct computation
- algebraic derivation
- backward verification
- decomposition
- approximation/bounding

目的：

在天然多解法任务上测试策略树指标。

