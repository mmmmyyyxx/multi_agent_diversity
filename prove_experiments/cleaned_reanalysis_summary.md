# 清洗后重分析摘要

本文记录本轮对 `prove_experiments` 的退化 trace 清洗检查与重新分析结论。

## 1. 污染是否变小

结论：变小了，但没有完全消失。

同口径扫描：

| 数据目录 | 退化题数 | 退化题率 | 退化 agent entry | entry 退化率 |
|---|---:|---:|---:|---:|
| 原始 `prove_experiments/runs` | 387 / 1200 | 32.25% | 1039 / 6000 | 17.32% |
| 清洗后 `prove_experiments/cleaned_runs` | 262 / 1200 | 21.83% | 662 / 6000 | 11.03% |

下降幅度：

- 退化题数下降约 32.30%。
- 退化 agent entry 下降约 36.28%。

## 2. 残余污染来源

清洗后残余污染主要集中在 qwen 和 gemini：

| 模型 | 清洗后退化题率 | 清洗后 entry 退化率 |
|---|---:|---:|
| `gpt-4o-mini` | 0.00% | 0.00% |
| `deepseek-chat` | 0.67% | 0.13% |
| `gemini-2.5-flash-lite` | 26.00% | 18.73% |
| `qwen2.5-7b-instruct` | 60.67% | 25.27% |

推荐解读：

- gpt/deepseek 子集基本干净，可信度较高。
- gemini/qwen 仍可用于观察趋势，但不适合单独支撑强结论。
- 四模型 P4 结果应被表述为“当前清洗后数据中的趋势”，而不是最终无污染结论。

## 3. 清洗后 P4 主结果

| 条件 | family_div | homogeneity | major_div | vote_acc |
|---|---:|---:|---:|---:|
| `mixed_strategy` | 0.5043 | 0.6576 | 0.4764 | 0.8250 |
| `same_definition` | 0.4765 | 0.6975 | 0.4329 | 0.8375 |
| `same_elimination` | 0.4399 | 0.7516 | 0.3569 | 0.8275 |

paired 统计：

- mixed - same 的 `family_div`：`+0.0277`，95% CI `[0.0044, 0.0507]`，Wilcoxon `p≈0.1087`。
- mixed - same 的 `homogeneity`：`-0.0399`，95% CI `[-0.0648, -0.0142]`，Wilcoxon `p≈0.0090`。
- mixed - same 的 `major_div`：`+0.0434`，95% CI `[0.0047, 0.0817]`，Wilcoxon `p≈0.0273`。

解释：

- 显式 mixed 策略 prompt 会提高策略多样性，并降低同质性。
- major-level 的提升比 family_div 的 Wilcoxon 检验更清楚。
- same_elimination 的 exact hit 高，是因为目标都集中在 option elimination，不能直接和 mixed 的 exact hit 难度等同。

## 4. 模型身份效应

P4 清洗后仍显示模型身份效应强于策略 prompt 效应：

- `strategy_effect_major_disagreement = 0.053`
- `model_identity_effect_major_disagreement = 0.268`
- `strategy_gt_model_identity = 0`

解释：

- 策略 prompt 有效，但不是主导全部差异。
- 跨模型时，策略树指标会混入模型身份、输出风格和 trace 质量差异。
- qwen/gemini 残余退化会进一步放大这种风险。

## 5. P6 / P8 清洗后结论

P6：

- `major-only` 太粗。
- `strict leaf-only` 太敏感。
- `weighted_tree` 是当前最合理的折中主指标，但不是已证明最优。

P8：

- `subject_count = 212`
- `paired_subject_count = 212`
- `mean_subject_intervention_effect = 0.0694`
- `positive_subject_rate = 0.6368`

解释：

- 多样性干预整体有正效应。
- 但不同 subject/model 组合差异很大。
- 不能只用总体均值，应报告 subject-level 分布。

## 6. 当前推荐主结论

当前最稳妥的结论是：

> 清洗后结果支持策略树指标能捕捉一部分真实策略差异。显式策略 prompt 会提高 family/major 层面的多样性并降低同质性。但该指标不是纯策略真值，会同时受到模型身份、输出风格、taxonomy 粒度、judge primary 偏差和退化 trace 影响。当前应优先使用 weighted-tree 与 major-level 指标，而不是 leaf exact hit；若要证明 reward 可优化性，仍需补跑 P5。

