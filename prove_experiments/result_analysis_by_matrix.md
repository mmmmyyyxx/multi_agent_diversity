# 证明实验结果分析（按实验矩阵 P1-P8）

本报告分析当前 `prove_experiments` 已完成的 non-P5 结果。总控状态为 `overall_status=ok`，覆盖：

- P2/P3/P4：4 个 solver model × 2 个 prompt 条件 × 100 道 MMLU 题。
- P1：200 条 trace，每条重复重判 3 次。
- P3 补充验证：40 条高风险 trace，由 `gpt-5.5` 分别做 taxonomy 内复判与 prompt-following 判断。
- P7：80 个匿名 trace group，由 `gpt-5.5` 盲评。
- P6/P8：离线重算与 subject-level 分析。

重要限制：

- P5 reward sweep 未运行，因此本报告不能回答“训练时该指标是否容易优化/是否过强约束”的最终问题。
- P4 中多数 run 是在 `family_rejudge_on_low_confidence=True` 时生成的，`qwen mixed` 是后续关闭该选项后生成的；严格论文结果建议统一配置重跑。
- P7 盲评结果与策略树指标明显不一致，这是当前最大负结果。
- P3 的目标策略命中解释已根据 `prove_experiments/p3_gpt55_combined_summary.md` 更新：优先采用 GPT-5.5 normal taxonomy judge 复判，prompt-following 只作为补充证据。

## 指标中文含义速查

本报告保留英文指标名，是为了能和 `csv/json/jsonl` 结果文件直接对应。下面给出每个指标的中文含义和读数方向。

通用实验字段：

| 指标 | 中文含义 | 读数方向 |
|---|---|---|
| `overall_status` | 一条龙实验脚本的总运行状态，`ok` 表示所有启用阶段完成。 | 只看是否为 `ok`。 |
| `model` | 生成推理 trace 的 solver 模型。 | 不是性能指标。 |
| `run` / `run_name` | 一次实验运行的名称，通常包含 P 编号、prompt 条件、模型和 seed。 | 不是性能指标。 |
| `bucket` | P7 抽样分桶，例如高策略树分、低策略树分、高文本差异低策略树分等。 | 用于构造对照组。 |
| `n` / `count` | 样本数。 | 越大统计越稳定。 |

策略树多样性相关指标：

| 指标 | 中文含义 | 读数方向 |
|---|---|---|
| `team_family_diversity` / `family_diversity` | 五个 agent 的 leaf 策略标签分布多样性，也就是“细粒度策略树多样性”。 | 越高表示策略标签越分散。 |
| `team_major_family_diversity` / `major_diversity` | 把 leaf 策略标签合并到 major family 后的多样性，也就是“粗粒度策略多样性”。 | 越高表示粗策略类别越分散。 |
| `team_family_homogeneity_rate` / `homogeneity` | 五个 agent 在策略标签上的同质性。 | 越高表示越像同一种策略；越低表示越多样。 |
| `team_intra_family_diversity` | 同一 major family 内部的 leaf 标签分散程度。 | 越高表示同一大类下面的细标签差异越多。 |
| `primary_family_labels` | 每个 agent 的主策略 leaf 标签。 | 用于人工检查标签来源。 |
| `secondary_family_labels` | 每个 agent 的次策略 leaf 标签。 | 用于观察混合策略或辅助策略。 |
| `primary_family_counts` | 五个 agent 的主策略标签计数。 | 看哪类策略占主导。 |
| `weighted_family_distribution` | 按主/次策略权重混合后的 leaf 策略分布。 | 是 weighted tree 计算的基础。 |
| `major_family_distribution` | 映射到 major family 后的策略分布。 | 是 major-only 多样性的基础。 |
| `all_same_primary` | 五个 agent 的主策略是否完全相同。 | `true` 表示主策略完全一致。 |
| `all_same_pair` | 五个 agent 的主/次策略组合是否完全相同。 | `true` 表示策略对完全一致。 |
| `primary_dominant_share` | 占比最高的主策略标签所占比例。 | 越高表示越被单一主策略支配。 |
| `pair_dominant_share` | 占比最高的主/次策略组合所占比例。 | 越高表示越被单一策略组合支配。 |

P1 judge 稳定性指标：

| 指标 | 中文含义 | 读数方向 |
|---|---|---|
| `trace_count` | 被重复重判的原始 trace 数量。 | 样本数。 |
| `judgment_count` | 总 judge 次数，通常等于 `trace_count × repeats`。 | 样本数。 |
| `mean_major_agreement` | 同一 trace 多次重判后，major family 标签的一致率均值。 | 越高说明粗标签越稳定。 |
| `mean_primary_agreement` | 同一 trace 多次重判后，primary leaf 标签的一致率均值。 | 越高说明主策略标签越稳定。 |
| `mean_pair_agreement` | 同一 trace 多次重判后，primary+secondary 标签对的一致率均值。 | 越高说明细粒度标签组合越稳定。 |
| `mean_confidence` | judge 输出的平均置信度。 | 越高表示 judge 自评越有把握。 |
| `major/primary/pair 完全一致率` | 所有重复重判都给出同一标签的 trace 比例。 | 越高越稳定。 |

P2/P3 prompt 干预指标：

| 指标 | 中文含义 | 读数方向 |
|---|---|---|
| `exact target hit` | agent 被要求使用某个目标策略时，judge 主/次标签是否精确命中目标 leaf。 | 是严格 taxonomy 对齐指标；偏低不必然等于模型完全不遵循。 |
| `same-major hit` | judge 标签是否命中目标策略所属的 major family。 | 比 exact target hit 更宽松，更适合判断是否落在目标策略附近。 |
| `GPT-5.5 taxonomy judge` | GPT-5.5 在同一 taxonomy 和正常 judge 信息下重新给 trace 贴策略标签。 | 用来检查原 judge/taxonomy 是否贴错标签。 |
| `primary/pair exact target hit` | GPT-5.5 taxonomy judge 的 primary 或 primary+secondary 是否精确命中目标 leaf。 | 严格检查目标 leaf 命中。 |
| `primary/pair same-major hit` | GPT-5.5 taxonomy judge 的 primary 或 primary+secondary 是否命中目标 major family。 | 检查是否在目标策略附近。 |
| `prompt-following followed_rate` | GPT-5.5 不看 taxonomy，只看原始策略指令和 trace，判断是否遵循目标策略的比例。 | 是外部语义补充证据，不能替代 taxonomy judge。 |
| `partial_or_better` | prompt-following 评分中 `adherence_score >= 3` 的比例。 | 表示至少有实质策略响应。 |
| `acc` / `vote_acc` | 五个 agent 投票后的答题准确率。 | 越高答题越准。 |
| `mean delta, mixed - same` | mixed 策略条件相对 same 策略条件的平均变化量。 | 正负取决于指标；diversity 希望为正，homogeneity 希望为负。 |
| `family_div delta` | mixed 相对 same 的细粒度策略树多样性变化。 | 越正表示显式策略干预越提高细粒度多样性。 |
| `homogeneity delta` | mixed 相对 same 的策略同质性变化。 | 越负表示策略更不相同。 |
| `major_div delta` | mixed 相对 same 的粗粒度策略多样性变化。 | 越正表示显式策略干预越提高粗策略多样性。 |
| `acc delta` | mixed 相对 same 的准确率变化。 | 接近 0 表示多样性变化不是靠准确率崩掉换来的。 |
| `95% bootstrap CI` | bootstrap 得到的 95% 置信区间。 | 区间不跨 0 表示方向较稳定。 |
| `Wilcoxon p approx` | 配对 Wilcoxon 检验的近似 p 值。 | 越小表示 mixed-same 差异越显著。 |

P4 跨模型迁移指标：

| 指标 | 中文含义 | 读数方向 |
|---|---|---|
| `strategy_effect_major_disagreement` | 同一模型内，same 与 mixed prompt 导致的 major family 分布差异。 | 越高表示策略 prompt 改变越大。 |
| `model_identity_effect_major_disagreement` | 同一 prompt 条件下，不同模型之间的 major family 分布差异。 | 越高表示模型身份/输出风格影响越大。 |
| `strategy_gt_model_identity` | 策略 prompt 效应是否大于模型身份效应。 | `1` 支持“主要测策略”；`0` 表示模型效应更大。 |
| `trace_token_div` | 五个 agent 原始 trace 的 token-cosine 多样性。 | 越高表示 trace 文本表面差异越大。 |
| `summary_token_div` | 五个 agent reasoning summary 的 token-cosine 多样性。 | 越高表示摘要文本差异越大。 |
| `trace_embedding_div` | 五个 agent 原始 trace 的 embedding-cosine 多样性，使用项目已有 embedding 指标。 | 越高表示语义向量差异越大。 |
| `summary_embedding_div` | 五个 agent reasoning summary 的 embedding-cosine 多样性。 | 越高表示摘要语义向量差异越大。 |

P5 reward sweep 相关指标：

| 指标 | 中文含义 | 读数方向 |
|---|---|---|
| `candidate family shift rate` | rewriter 产生的新候选 prompt 是否改变了候选 trace 的策略标签分布。 | 越高表示搜索能探索到不同策略。 |
| `update applied rate` | 训练中候选 prompt 被接受并更新的比例。 | 太低可能说明 reward 太难优化或搜索弱。 |
| `invalid trace` / `invalid trace rate` | 无效、乱码、无法解析、answer-only 等低质量 trace 的比例。 | 越低越好；升高说明 reward 可能诱导坏行为。 |
| `softened_tree` | 较宽松的策略树 reward 版本，通常降低细粒度惩罚。 | 用来测试约束是否过强。 |
| `strict_tree` | 更严格的 leaf-level 策略树 reward 版本。 | 用来测试细粒度约束是否导致训练困难。 |

P6 taxonomy 粒度指标：

| 指标 | 中文含义 | 读数方向 |
|---|---|---|
| `major-only` | 只看 major family 的粗粒度多样性。 | 较稳健，但可能漏掉细策略差异。 |
| `weighted tree` | 当前主用指标，综合主/次 leaf 策略分布和层级权重。 | 理想上兼顾粗细粒度。 |
| `strict leaf` | 只看 leaf 标签的严格细粒度多样性。 | 可能更敏感，也更容易把表述差异误判为策略差异。 |
| `run 内相关` | 同一个 run 内，某个策略树指标和 GPT-5.5 盲评分数的相关性。 | 越接近正相关越支持该粒度。 |

P7 GPT-5.5 盲评指标：

| 指标 | 中文含义 | 读数方向 |
|---|---|---|
| `sampled_groups` | P7 盲评的匿名 trace 组数量，每组通常包含五个 agent trace。 | 样本数。 |
| `mean_gpt_method_diversity_score` | GPT-5.5 给出的真实方法多样性平均分，范围 1-5。 | 越高表示 GPT 认为方法越多样。 |
| `gpt_distinct_methods_count` | GPT-5.5 判断一组 trace 中有多少种不同方法。 | 越高表示方法数越多。 |
| `gpt_confidence` | GPT-5.5 对自己盲评判断的置信度。 | 越高表示自评越有把握。 |
| `strategy_tree_vs_gpt_spearman` | 策略树细粒度多样性与 GPT-5.5 盲评分数的 Spearman 秩相关。 | 越接近 +1 越一致；接近 0 表示几乎无单调关系。 |
| `major_tree_vs_gpt_spearman` | major-only 多样性与 GPT-5.5 盲评分数的 Spearman 秩相关。 | 越接近 +1 越一致。 |
| `trace_text_vs_gpt_spearman` | 原始 trace 文本差异与 GPT-5.5 盲评分数的 Spearman 秩相关。 | 越高说明 GPT 更像在跟随文本差异。 |
| `high_strategy - low_strategy GPT score` | 高策略树分桶与低策略树分桶的 GPT 平均分差。 | 越正越支持策略树指标；接近 0 表示不支持。 |
| `GPT mean` | 某个 bucket 内 GPT-5.5 盲评分数均值。 | 越高表示该 bucket 真实方法更多样。 |
| `tree mean` | 某个 bucket 内策略树细粒度多样性均值。 | 越高表示自动策略标签越分散。 |
| `major mean` | 某个 bucket 内 major-only 多样性均值。 | 越高表示粗策略类别越分散。 |
| `text mean` | 某个 bucket 内 trace 文本多样性均值。 | 越高表示文本表面差异越大。 |

P8 任务依赖指标：

| 指标 | 中文含义 | 读数方向 |
|---|---|---|
| `唯一 subject 数` | MMLU 中出现的学科/子任务数量。 | 样本覆盖度。 |
| `subject-model paired rows` | subject 与 model 组合后，能配对比较 same/mixed 的单元数。 | 样本数。 |
| `mean_subject_intervention_effect` | 按 subject-model 单元计算，mixed 相对 same 的平均策略多样性提升。 | 越正表示任务内策略干预更有效。 |
| `median effect` | subject-model 单元干预效应的中位数。 | 比均值更不受极端值影响。 |
| `positive_subject_rate` | mixed 多样性高于 same 的 subject-model 单元比例。 | 越高说明更多任务上干预有效。 |

## P1. Judge 可靠性与标签稳定性

实验目标：验证同一条完整 trace 多次输入策略树 judge 时，标签是否稳定。

结果：

| 指标 | 当前结果 | 预设通过线 |
|---|---:|---:|
| trace_count | 200 | - |
| judgment_count | 600 | - |
| mean_major_agreement | 0.9700 | 0.85 |
| mean_primary_agreement | 0.9533 | 0.70 |
| mean_pair_agreement | 0.8617 | 高于随机 |
| mean_confidence | 0.8433 | - |

分布补充：

- major 完全一致率：0.915。
- primary 完全一致率：0.865。
- pair 完全一致率：0.630。
- primary/pair 的低尾部样本存在，最低一致率为 0.3333，但均值和 10 分位数仍较高。

解释：

P1 强支持“judge 本身不是随机漂移”的前提。尤其是 primary 和 major 的一致率都显著超过预设标准，说明后续 P2/P3/P4 观察到的差异不太可能只是同一 trace 重判噪声造成的。

但 P1 只能证明同一 judge 对同一 trace 稳定，不能证明标签语义一定等于真实策略差异。这个问题要看 P7。

## P2. 同策略改写负对照

实验目标：5 个 agent 都被要求使用 broad option elimination，只是 prompt 措辞不同；理想指标应避免把 paraphrase 当成高策略多样性。

run 级结果：

| model | family_diversity | homogeneity | major_diversity | exact target hit | same-major hit | acc |
|---|---:|---:|---:|---:|---:|---:|
| deepseek-chat | 0.4773 | 0.6936 | 0.4822 | 0.5660 | 0.5900 | 0.9000 |
| gemini-2.5-flash-lite | 0.3873 | 0.8395 | 0.2829 | 0.7580 | 0.7660 | 0.7900 |
| gpt-4o-mini | 0.4219 | 0.7635 | 0.3486 | 0.7760 | 0.8040 | 0.8400 |
| qwen2.5-7b-instruct | 0.5121 | 0.6402 | 0.3454 | 0.6460 | 0.7860 | 0.7200 |

整体均值：

- mean_family_diversity：0.4496。
- mean_homogeneity：0.7342。
- mean_major_diversity：0.3648。
- exact target hit：0.6865。
- same-major target hit：0.7365。
- vote_acc：0.8125。

解释：

P2 有中等支持：同策略条件下，target hit rate 较高，说明 judge 大体能识别“这些 trace 接近 option elimination”。但 same 条件的 diversity 并不低，特别是 qwen 和 deepseek 的 same diversity 偏高。

这说明负对照没有完全干净。可能原因有三类：

- 同一个 option-elimination prompt 在不同底座模型上会诱导不同实际 trace 形态。
- MMLU 题目本身会让 agent 混入 rule application、direct recall、causal reasoning 等元素。
- 当前 taxonomy 粒度偏细，会把同一底层方法的不同表述拆成多个 leaf label。

P2 的结论应写成：指标能部分避免纯 prompt paraphrase 伪差异，但仍存在同策略下分数偏高的问题。

## P3. 显式策略干预

实验目标：显式要求 5 个 agent 使用不同策略后，策略树多样性应高于 P2 同策略负对照。

整体 paired 结果：

| 对比 | mean delta, mixed - same | 95% bootstrap CI | Wilcoxon p approx |
|---|---:|---:|---:|
| team_family_diversity | +0.0650 | [0.0445, 0.0856] | 2.0e-8 |
| team_family_homogeneity_rate | -0.1019 | [-0.1282, -0.0759] | 2.9e-13 |
| team_major_family_diversity | +0.1325 | [0.0905, 0.1719] | 8.7e-10 |

各模型 mixed - same：

| model | family_div delta | homogeneity delta | major_div delta | acc delta |
|---|---:|---:|---:|---:|
| deepseek-chat | +0.1048 | -0.1859 | +0.2218 | -0.0200 |
| gemini-2.5-flash-lite | +0.0308 | -0.0168 | +0.0365 | +0.0100 |
| gpt-4o-mini | +0.0586 | -0.0640 | +0.0610 | +0.0200 |
| qwen2.5-7b-instruct | +0.0656 | -0.1410 | +0.2108 | -0.0100 |

解释：

P3 强支持“策略树指标对显式策略 prompt 干预敏感”。四个模型方向一致，统计显著，且准确率整体没有变化：

- same 平均 acc：0.8125。
- mixed 平均 acc：0.8125。

这排除了“多样性提升只是因为整体答题质量崩掉”的简单解释。

但 P3 的目标策略命中率出现一个反直觉现象：

- same exact target hit：0.6865。
- mixed exact target hit：0.2530。
- same-major hit：same 0.7365，mixed 0.6055。

进一步拆解见 `prove_experiments/p3_target_compliance/p3_target_compliance.md`。随后又做了 GPT-5.5 补充验证，综合见 `prove_experiments/p3_gpt55_combined_summary.md`。最新解释要分两层：

第一层是主证据：GPT-5.5 作为 normal taxonomy judge，在完整 taxonomy、major-family tree、family definitions 和正常 judge 同等信息下复判 40 条高风险样本。这些样本的 target 不包含 `option_contrast`，但原自动 judge primary 全部是 `option_contrast`。结果是：

| 指标 | 结果 | 含义 |
|---|---:|---|
| GPT primary option | 0.0750 | 只有 3/40 被 GPT-5.5 也判为 primary `option_contrast`。 |
| GPT pair option | 0.1750 | 只有 7/40 在 primary 或 secondary 中含 `option_contrast`。 |
| original judge supported | 0.0750 | 原 `option_contrast` primary 只有很少被支持。 |
| judge/taxonomy questioned | 0.9250 | 大多数原 `option_contrast` primary 被质疑。 |
| GPT confidence | 0.8960 | GPT-5.5 对复判较有把握。 |

这说明 mixed exact target hit 低，首先应被解释为 **原 judge/taxonomy 对 `option_contrast` 有明显吸附，并且 primary/secondary 排序存在问题**。它不是简单的模型不遵循策略。

第二层仍然来自 GPT-5.5 taxonomy judge：GPT-5.5 不支持原 `option_contrast`，但也没有把大多数 trace 精确贴回目标 leaf。

| 指标 | 结果 | 含义 |
|---|---:|---|
| primary exact target hit | 0.1250 | GPT-5.5 primary 精确命中目标 leaf 的比例仍低。 |
| pair exact target hit | 0.1250 | primary 或 secondary 精确命中目标 leaf 的比例仍低。 |
| primary same-major hit | 0.6000 | GPT-5.5 primary 落在目标 major family 的比例较高。 |
| pair same-major hit | 0.7000 | primary 或 secondary 落在目标 major family 的比例较高。 |

因此，更准确的判断是：原 judge 的 `option_contrast` primary 多数不成立；GPT-5.5 认为很多 trace 位于目标策略附近的大类；但 leaf exact target hit 仍然过严，受 taxonomy 边界、相邻 leaf 和主次标签排序影响。

第三层是补充证据：GPT-5.5 不看 taxonomy，只看原始策略指令和 trace 做 prompt-following 判断。结果：

| 指标 | 结果 | 含义 |
|---|---:|---|
| followed_rate | 0.7250 | 29/40 被认为遵循目标策略。 |
| partial_or_better | 0.8750 | 35/40 至少部分遵循。 |
| mean_score | 3.9000 | 平均遵循分数接近 4/5。 |
| judge_taxonomy_likely | 0.7000 | 多数更像 judge/taxonomy 贴标问题。 |
| model_prompt_likely | 0.1250 | 少数更像模型/prompt 没有稳定诱导目标策略。 |
| ambiguous | 0.1750 | 一部分边界模糊。 |

这个补充实验不能替代 taxonomy judge，但支持上述解释：低 exact target hit 不能直接等价于“模型完全不听策略 prompt”。

综合看，mixed target hit 低是三种因素叠加：

1. same 条件的目标本来更容易命中。same-elimination 的五个 agent 都以 `distractor_elimination\|option_contrast` 为核心目标，而这正是 MMLU 多选题最自然、最常见的 trace 形态。
2. 原 judge/taxonomy 对 `option_contrast` 有明显吸附，容易把“逐项检查选项”的表面形式提升成 primary。
3. mixed 条件的目标策略难度差异很大；`rule_or_principle_application` 等目标确实存在 prompt 可执行性或模型遵循不足。

mixed 条件按目标策略拆开后：

| agent | 目标策略 | 原 judge exact hit | 原 judge same-major hit | top primary label | 最新解读 |
|---|---|---:|---:|---|---|
| 0 | `concept_definition_match` | 0.1400 | 0.6450 | `option_contrast` 0.4525 | GPT-5.5 taxonomy 复判显示 pair exact target hit 为 0.5000，prompt-following followed 为 0.7000；原 judge 的 `option_contrast` 更像过度吸附。 |
| 1 | `distractor_elimination\|option_contrast` | 0.6600 | 0.7075 | `option_contrast` 0.5500 | 遵循最好。说明模型和 judge 对“逐项比较/排除”这种策略比较一致。 |
| 2 | `answer_to_stem_backward_check\|option_contradiction_check` | 0.1750 | 0.7375 | `option_contrast` 0.6100 | GPT-5.5 taxonomy exact 为 0，但 same-major 为 0.9000；prompt-following followed 为 0.9000。说明它常落在 option-semantics 大类内，但 leaf 边界与 `option_contrast` 混淆严重。 |
| 3 | `rule_or_principle_application` | 0.0375 | 0.1800 | `option_contrast` 0.3675 | 最弱。GPT-5.5 taxonomy exact 与 same-major 都为 0，prompt-following followed 也只有 0.5000。这里确实存在 prompt 可诱导性或模型遵循问题。 |
| 4 | `decomposition\|stem_evidence_alignment` | 0.2525 | 0.7575 | `option_contrast` 0.3750 | GPT-5.5 taxonomy exact 为 0，但 same-major 为 1.0000，prompt-following followed 为 0.8000。更像相近策略/major 命中被 leaf exact 低估。 |

如果把 secondary 的 major family 也纳入，原统计口径下 mixed 的 same-major 命中会从 0.6055 提高到约 0.6535。GPT-5.5 taxonomy judge 的高风险样本中 pair same-major hit 进一步达到 0.7000。这说明一部分“未命中”不是完全跑偏，而是 primary/secondary 归属或 leaf 边界造成的。

按模型看，模式也不是均匀的：

- `distractor_elimination\|option_contrast` 在四个模型上都相对高：deepseek 0.52，gemini 0.77，gpt-4o-mini 0.74，qwen 0.61。
- `rule_or_principle_application` 在四个模型上都极低：0.03-0.05 左右，same-major 也只有 0.16-0.19。这更像目标策略本身在当前 MMLU prompt/trace 格式下不可稳定诱导，不能只怪某一个模型。
- `decomposition\|stem_evidence_alignment` 模型差异很大：deepseek exact 0.55，但 gemini 0.06，gpt-4o-mini 0.11，qwen 0.29。这提示模型输出风格会显著影响 target hit。
- `answer_to_stem_backward_check\|option_contradiction_check` 的 same-major 高但 exact 低，说明它多数仍被吸到 option-semantics 大类里，但没有形成清晰的 backward-check leaf。

因此 P3 的正确解释应改为：P3 强证明“显式 mixed prompt 能改变自动标签分布，并提高策略树多样性”；GPT-5.5 taxonomy 复判进一步证明，原 judge 的 `option_contrast` primary 多数站不住；prompt-following 补充评审说明，多数高风险 trace 仍有真实策略响应。但 P3 不能强证明“五个 agent 都严格按照指定 leaf 策略执行”。mixed exact target hit 低同时暴露了 judge/taxonomy leaf 边界、primary/secondary 排序、模型策略遵循和 prompt 可诱导性四个问题。

潜在风险：

- 如果把 `target_exact_hit_rate` 当作模型遵循能力的唯一指标，会低估相近策略、same-major 策略或 secondary 策略中的真实遵循。
- 如果把 P3 diversity 提升直接解释成“真实策略都按要求改变了”，会被 P7 的盲评反例削弱。
- 如果后续训练 reward 强行优化 exact leaf 命中，模型可能学会堆关键词、套格式，甚至牺牲有效解题，而不是真的发展出不同方法。
- 对 `rule_or_principle_application` 这类当前几乎不可精确命中的策略，严格 leaf reward 可能过强，导致训练信号稀疏。
- mixed 条件各目标策略难度不同，直接平均 exact hit 会让“难策略”拉低整体结果，造成跨策略比较偏差。

建议后续把 target compliance 拆成三层报告，并在论文里优先解释前两层：

- leaf exact compliance：是否精确命中目标 leaf。
- GPT-5.5 normal taxonomy compliance：在同一 taxonomy 和正常 judge 信息下，复判是否仍支持原标签，以及是否命中目标 leaf/major。
- major compliance：是否落入目标大类，尤其同时看 primary 和 secondary。
- independent prompt-following compliance：作为外部补充，让 GPT-5.5 只根据原始策略指令和 trace 判断“是否真的按这个策略执行”，不使用策略树标签。

## P4. 跨 LLM 策略迁移

实验目标：判断策略树测到的是策略 prompt 效应，还是模型身份/风格效应。

模型集合：

- `gpt-4o-mini`
- `gemini-2.5-flash-lite`
- `deepseek-chat`
- `qwen2.5-7b-instruct`

核心结果：

| 指标 | 值 |
|---|---:|
| strategy_effect_major_disagreement | 0.1030 |
| model_identity_effect_major_disagreement | 0.2135 |
| strategy_gt_model_identity | 0 |

补充多样性指标：

项目已有 `scripts/compute_experiment_metrics.py` 会计算 prompt/trace/summary 的 token-cosine 与 embedding-cosine 多样性。本轮已用同一脚本对 P4 的 8 个 run 重新计算，输出见 `prove_experiments/p4_embedding_metrics.csv` 和 `prove_experiments/p4_embedding_metrics.md`。embedding 模型状态为 `ok`。

各模型、各条件的文本与 embedding 多样性：

| model | condition | trace token div | trace embedding div | summary token div | summary embedding div |
|---|---|---:|---:|---:|---:|
| deepseek-chat | mixed | 0.1763 | 0.0444 | 0.1831 | 0.0596 |
| gemini-2.5-flash-lite | mixed | 0.0787 | 0.0300 | 0.1342 | 0.0402 |
| gpt-4o-mini | mixed | 0.1409 | 0.0454 | 0.1794 | 0.0575 |
| qwen2.5-7b-instruct | mixed | 0.6139 | 0.2560 | 0.2755 | 0.1631 |
| deepseek-chat | same | 0.1465 | 0.0371 | 0.1696 | 0.0519 |
| gemini-2.5-flash-lite | same | 0.0742 | 0.0277 | 0.1307 | 0.0379 |
| gpt-4o-mini | same | 0.1126 | 0.0330 | 0.1698 | 0.0516 |
| qwen2.5-7b-instruct | same | 0.5878 | 0.2473 | 0.2463 | 0.1573 |

策略效应与模型身份效应对比：

| metric | mean mixed - same | strategy effect abs | model identity effect abs | strategy > model |
|---|---:|---:|---:|---:|
| family diversity | +0.0650 | 0.0650 | 0.0850 | 0 |
| major diversity | +0.1325 | 0.1325 | 0.1584 | 0 |
| homogeneity | -0.1019 | 0.1019 | 0.1525 | 0 |
| trace token div | +0.0221 | 0.0221 | 0.2680 | 0 |
| summary token div | +0.0140 | 0.0140 | 0.0645 | 0 |
| trace embedding div | +0.0077 | 0.0077 | 0.1118 | 0 |
| summary embedding div | +0.0054 | 0.0054 | 0.0608 | 0 |

解释：

P4 是负面或至少混合结果。虽然每个模型内部 mixed 都高于 same，但跨模型差异比策略 prompt 差异更大。这个结论不只出现在 major-family disagreement 上，也出现在多种多样性指标上：

- 策略树 family diversity 的 mixed-same 平均提升为 +0.0650，但模型身份效应为 0.0850。
- major diversity 的 mixed-same 平均提升为 +0.1325，但模型身份效应为 0.1584。
- trace token 多样性的 mixed-same 平均提升只有 +0.0221，但模型身份效应达到 0.2680。
- trace embedding 多样性的 mixed-same 平均提升只有 +0.0077，但模型身份效应达到 0.1118。
- summary embedding 多样性的 mixed-same 平均提升为 +0.0054，模型身份效应为 0.0608。

这意味着当前指标受到模型身份影响：不同模型即使在同一 broad strategy 下，也可能产生不同标签分布、文本差异和 embedding 差异。原因可能是：

- 模型输出风格不同，judge 把风格差异解释成策略差异。
- 开源/中转模型有更多退化或短答 trace，改变了标签分布。
- MMLU 多选题上底层解法空间有限，模型差异反而更显眼。
- qwen2.5-7b-instruct 的 trace token/embedding 多样性远高于其他模型，说明部分跨模型差异可能来自输出长度、格式、退化行为或可见推理量，而不只是策略差异。

P4 不能支持“指标主要测策略而不是模型风格”。当前应写成：策略 prompt 有稳定方向效应，但模型身份效应在策略树、文本 token 和 embedding 指标上都更大，需要进一步控制。更严格的后续做法是：在同一模型内报告策略效应，在跨模型分析时控制输出长度/invalid/answer-only/summary 质量，并使用统一 judge 配置重跑。

## P5. Reward 权重 sweep

当前未运行。

因此不能得出以下结论：

- 策略树 reward 是否可优化。
- 指标是否过强导致训练难提升。
- softened/strict tree 哪个更适合训练。
- candidate family shift rate、update applied rate、invalid trace 是否支持训练信号。

本轮结果只覆盖“测量有效性”的一部分，不覆盖“作为训练 reward 的可优化性”。

## P6. Taxonomy 粒度敏感性

实验目标：比较 major-only、当前 weighted tree、strict leaf-only 三种粒度。

主要结果：

| run | major-only | weighted tree | strict leaf |
|---|---:|---:|---:|
| mixed deepseek | 0.7181 | 0.5821 | 0.7845 |
| mixed gemini | 0.2748 | 0.4180 | 0.3973 |
| mixed gpt4omini | 0.3987 | 0.4805 | 0.6156 |
| mixed qwen | 0.5683 | 0.5777 | 0.8673 |
| same deepseek | 0.4583 | 0.4773 | 0.5219 |
| same gemini | 0.2415 | 0.3873 | 0.3408 |
| same gpt4omini | 0.3510 | 0.4219 | 0.4347 |
| same qwen | 0.3647 | 0.5121 | 0.7469 |

带 GPT-5.5 盲评分数重算后，weighted tree 与 GPT 分数的 run 内相关多数接近 0 或为负：

- mixed deepseek：+0.0501。
- mixed gemini：-0.3261。
- mixed gpt4omini：-0.1551。
- mixed qwen：-0.3123。
- same deepseek：+0.0858。
- same gemini：-0.0168。
- same gpt4omini：-0.1183。
- same qwen：-0.1823。

解释：

P6 不支持“当前 weighted tree 与独立构念判断最一致”。strict leaf 往往更高，但可能明显过敏；major-only 有时更低，有时反而高于 weighted tree。

结合 P7 示例看，问题很可能是粒度错配：

- 对数学/物理题，`decomposition`、`equation_solving`、`algebraic_derivation`、`direct_computation` 可能只是同一公式解法的不同描述。
- 对退化 trace，策略树可能没有把 answer-only/乱码/无效输出作为单独行为维度处理。

因此当前 taxonomy 更适合作为“标签分散度诊断”，还不能直接等同于“真实方法多样性”。

## P7. GPT-5.5 盲评验证

实验目标：独立 GPT-5.5 只看匿名 trace，判断高策略树多样性的组是否真的方法更多样。

结果：

| 指标 | 值 |
|---|---:|
| sampled_groups | 80 |
| mean_gpt_method_diversity_score | 1.5125 |
| strategy_tree_vs_gpt_spearman | 0.0075 |
| major_tree_vs_gpt_spearman | -0.0592 |
| trace_text_vs_gpt_spearman | 0.6829 |
| high_strategy - low_strategy GPT score | -0.0250 |
| 95% CI | [-0.3000, 0.2750] |

GPT 分数分布：

| score | count |
|---:|---:|
| 1 | 46 |
| 2 | 28 |
| 3 | 5 |
| 4 | 1 |

按 bucket：

| bucket | n | GPT mean | tree mean | major mean | text mean |
|---|---:|---:|---:|---:|---:|
| high_strategy | 20 | 1.8 | 0.7445 | 0.8697 | 0.3986 |
| low_strategy | 20 | 1.25 | 0.2623 | 0.0722 | 0.2116 |
| high_text_low_strategy | 20 | 1.8 | 0.2962 | 0.0325 | 0.5217 |
| low_text_high_strategy | 20 | 1.2 | 0.6878 | 0.7973 | 0.0672 |

解释：

P7 是当前最重要的反证。GPT-5.5 盲评几乎不认为策略树分数高就代表真实方法更多样；它更接近文本差异。

代表性样本：

- 策略树高但 GPT 低：水平抛运动题，标签分成 `decomposition`、`equation_solving`、`algebraic_derivation`，但五条 trace 都是同一个公式 `y = 1/2 g t^2`，GPT 给 1/5。
- 策略树高但 GPT 低：Newton 第三定律题，标签有 `rule_or_principle_application`、`causal_mechanism_reasoning`、`answer_to_stem_backward_check`，但五条 trace 都是套牛顿第三定律，GPT 给 1/5。
- 策略树低但 GPT 较高：Euthyphro 题，标签几乎全是 `option_contrast`，但 trace 实际包含完整选项排除、浅层 all-of-the-above 判断、answer-only 三种行为，GPT 给 3/5。

这说明 GPT 的“方法多样性”包含：

- 是否 answer-only 或退化。
- 推理深度。
- 信息来源和核心知识是否不同。
- 是否只是同一公式/同一事实的不同包装。

而策略树更关注 leaf label 差异，可能把同一底层方法拆得过细，也可能忽略输出行为质量差异。

## P8. 任务依赖检查

实验目标：判断 MMLU subject 对 reachable strategy diversity 的影响。

结果：

- 唯一 subject 数：53。
- subject-model paired rows：212。
- mean_subject_intervention_effect：0.0706。
- median effect：0.0552。
- positive_subject_rate：0.6226。

解释：

P8 支持“任务依赖存在”。大约 62% 的 subject-model 单元中，mixed prompt 的策略树 diversity 高于 same prompt。但很多 subject 在 100 题采样里只有 1-3 道题，所以 subject-level CI 很宽，不能过度解释具体学科排序。

这个结果更适合作为一个提醒：

- MMLU 一些题本身只支持一个主导方法，例如直接事实 recall、固定公式、单一规则应用。
- 在这些题上，即使 prompt 要求不同策略，真实 trace 也可能收敛到同一方法。
- 因此策略多样性指标应与任务可达策略空间一起报告，而不是假设每道题都应高多样。

## 总体结论

当前结果支持三点：

1. 策略树 judge 稳定。
2. 显式 mixed strategy prompt 会显著提高自动 strategy-tree diversity。
3. 这种提升跨 4 个 solver 都存在，且不依赖准确率下降。
4. P3 的 GPT-5.5 normal taxonomy 复判显示，原 judge 对 `option_contrast` primary 存在明显吸附；GPT-5.5 prompt-following 补充评审显示，多数高风险 trace 仍有真实策略响应信号。

当前结果不支持或削弱三点：

1. P7 不支持“strategy-tree diversity 等于 GPT-5.5 眼中的真实方法多样性”。
2. P4 显示模型身份效应大于策略 prompt 效应。
3. P6 不支持当前 weighted taxonomy 是最优粒度。
4. P3 不支持“mixed 条件下五个 agent 都精确命中指定 leaf 策略”，leaf exact target hit 仍然很低。

因此最准确的表述是：

> 当前策略树指标是一个稳定、对 prompt 干预敏感的自动标签分散度指标；P3 的 GPT-5.5 normal taxonomy 复判进一步说明，原 judge 的 `option_contrast` primary 会系统性低估 mixed 策略响应。但它尚未被证明能充分测量独立评估意义上的真实推理方法差异。主要问题是 taxonomy 粒度与真实方法构念不完全对齐，`option_contrast` 等 leaf 存在吸附和主次排序问题，并且模型身份/输出风格会影响标签分布。

## 建议下一步

1. 重新定义 P7 的评价维度，把“answer-only/invalid/乱码”与“有效策略差异”分开评分。
2. 合并数学/物理中的近邻 leaf：如 `equation_solving`、`direct_computation`、`algebraic_derivation` 在很多 MMLU 题上应归为同一底层方法。
3. 增加 strategy-tree 的 validity-aware 版本：先过滤 invalid/answer-only，再计算有效策略多样性。
4. 收紧 `option_contrast` 判定：只有当“选项之间的相互比较”是主导组织结构时才作为 primary；若 trace 先建立概念、规则、证据或机制，再检查选项，应把 `option_contrast` 放 secondary 或不贴。
5. 重写 `rule_or_principle_application` prompt，要求模型显式输出“规则/原则是什么”和“该规则如何约束 stem”，再进入选项。
6. 统一 `family_rejudge_on_low_confidence` 配置后重跑 P2/P3/P4，避免配置混杂。
7. 跑 P5 sweep，单独回答 reward 是否太强、是否可优化。
