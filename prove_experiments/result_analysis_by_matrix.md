# 证明实验结果分析（按实验矩阵 P1-P8）

本报告分析当前 `prove_experiments` 已完成的 non-P5 结果。总控状态为 `overall_status=ok`，覆盖：

- P2/P3/P4：4 个 solver model × 2 个 prompt 条件 × 100 道 MMLU 题。
- P1：200 条 trace，每条重复重判 3 次。
- P7：80 个匿名 trace group，由 `gpt-5.5` 盲评。
- P6/P8：离线重算与 subject-level 分析。

重要限制：

- P5 reward sweep 未运行，因此本报告不能回答“训练时该指标是否容易优化/是否过强约束”的最终问题。
- P4 中多数 run 是在 `family_rejudge_on_low_confidence=True` 时生成的，`qwen mixed` 是后续关闭该选项后生成的；严格论文结果建议统一配置重跑。
- P7 盲评结果与策略树指标明显不一致，这是当前最大负结果。

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

原因是 same 条件所有 agent 的目标都集中在 option elimination，比较容易命中；mixed 条件每个 agent 的目标更细、更分散，当前 judge/taxonomy 对精确策略遵循的捕捉并不强。因此 P3 证明的是“标签分布改变了”，不是“每个 agent 都严格执行了指定策略”。

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

解释：

P4 是负面或至少混合结果。虽然每个模型内部 mixed 都高于 same，但跨模型差异比策略 prompt 差异更大。

这意味着当前指标受到模型身份影响：不同模型即使在同一 broad strategy 下，也可能产生不同标签分布。原因可能是：

- 模型输出风格不同，judge 把风格差异解释成策略差异。
- 开源/中转模型有更多退化或短答 trace，改变了标签分布。
- MMLU 多选题上底层解法空间有限，模型差异反而更显眼。

P4 不能支持“指标主要测策略而不是模型风格”。当前应写成：策略 prompt 有稳定方向效应，但模型身份效应仍然更大，需要进一步控制。

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

当前结果不支持或削弱三点：

1. P7 不支持“strategy-tree diversity 等于 GPT-5.5 眼中的真实方法多样性”。
2. P4 显示模型身份效应大于策略 prompt 效应。
3. P6 不支持当前 weighted taxonomy 是最优粒度。

因此最准确的表述是：

> 当前策略树指标是一个稳定、对 prompt 干预敏感的自动标签分散度指标；但它尚未被证明能充分测量独立评估意义上的真实推理方法差异。主要问题是 taxonomy 粒度与真实方法构念不完全对齐，并且模型身份/输出风格会影响标签分布。

## 建议下一步

1. 重新定义 P7 的评价维度，把“answer-only/invalid/乱码”与“有效策略差异”分开评分。
2. 合并数学/物理中的近邻 leaf：如 `equation_solving`、`direct_computation`、`algebraic_derivation` 在很多 MMLU 题上应归为同一底层方法。
3. 增加 strategy-tree 的 validity-aware 版本：先过滤 invalid/answer-only，再计算有效策略多样性。
4. 统一 `family_rejudge_on_low_confidence` 配置后重跑 P2/P3/P4，避免配置混杂。
5. 跑 P5 sweep，单独回答 reward 是否太强、是否可优化。
