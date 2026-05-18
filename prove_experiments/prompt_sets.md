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

- `distractor_elimination`
- major family：`mmlu_option_semantics`

Prompt：

Use a distractor-elimination strategy. Treat each answer choice as a candidate, reject choices that conflict with the stem, and keep the best-supported remaining choice. End with exactly one `FINAL_ANSWER:` line.

Agent 1 target：

- `rule_or_principle_application`
- major family：`mmlu_domain_reasoning`

Prompt：

Use a domain-rule strategy. First state the governing rule, theorem, principle, mechanism, or domain law, then apply that rule to the facts in the stem before choosing. End with exactly one `FINAL_ANSWER:` line.

Agent 2 target：

- `decomposition`
- major family：`representation_formalization`

Prompt：

Use a decomposition strategy. Break the stem into key facts, constraints, and sub-questions, solve each part in order, and combine the parts into one answer. End with exactly one `FINAL_ANSWER:` line.

Agent 3 target：

- `case_analysis`
- major family：`logical_proof`

Prompt：

Use a case-analysis strategy. Enumerate the relevant cases, conditions, or scenarios implied by the stem, test each case for consistency, and choose the answer that survives the case split. End with exactly one `FINAL_ANSWER:` line.

Agent 4 target：

- `edge_case_analysis`
- major family：`optimization_boundary_meta`

Prompt：

Use an edge-case and exception-checking strategy. Look for boundary conditions, qualifiers, extreme cases, or exceptions in the stem, then decide which answer remains valid under those checks. End with exactly one `FINAL_ANSWER:` line.

期望结果：

family diversity 和 major diversity 应高于 P2，同质性应降低；目标 leaf exact hit 只作为诊断项，因为它会受到 leaf 粒度、MMLU 多选题形态和 judge/taxonomy primary 判定规则影响。

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

