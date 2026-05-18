# 分析计划

本文档定义 `prove_experiments` 的分析方式。建议在查看结果前固定这些标准，避免事后解释。

## 分析单位

Trace 级：

- 一个 agent 在一道题上的完整推理轨迹。
- 用于标签稳定性、instructed-family hit rate 和 GPT-5.5 盲评一致性。

Question-level team 级：

- 同一道题上 5 个 agent 的 trace 组。
- 用于 team family diversity、homogeneity、all-same pair rate 和 GPT-5.5 method-diversity 评分。

Run 级：

- train、validation 或 test 上的平均指标。
- 用于 reward 可优化性和早停比较。

## 优先使用的现有指标

先使用框架已有指标：

- `team_family_diversity`
- `team_family_homogeneity_rate`
- `all_same_pair`
- `primary_family_labels`
- `secondary_family_labels`
- `family_confidences`
- `low_confidence_share`
- `rejudge_count`
- `disagreement_rate`
- `prompt_drift_cosine_distance`
- `update_applied_rate`
- candidate diagnostics 中的 `family_shift_rate`
- 可用时的 `invalid_trace_penalty`
- `trace_embedding_cosine_diversity`

除非证明实验显示现有指标无法诊断关键失败模式，否则不要新增 reward 指标。

## 仅用于证明分析的诊断量

这些量只用于分析，不进入训练 reward。

### Instructed-Family Hit Rate

定义：

对于目标 family 集合 `T_i` 的 agent，如果满足下面任一条件，就记为命中：

`primary_family in T_i`，或 `secondary_family in T_i`，或 `major(primary_family) == major(any T_i)`。

计算：

`hit_rate_i = hits_i / total_traces_i`

物理意义：

衡量显式策略指令是否真的把模型推理行为推向目标方向。

注意：

P3 中应同时报告 exact hit 和 same-major hit，但不要把每个 leaf 的 exact hit 设为硬性通过阈值。exact hit 会受到 taxonomy 粒度、MMLU 多选题形态和 judge primary 选择规则影响；它更适合用于诊断哪些策略 prompt 难遵循、哪些标签存在吸附，而不是单独决定指标是否有效。

### Strategy Intervention Effect

定义：

`intervention_effect = mean_diversity(mixed_strategy_prompts) - mean_diversity(same_strategy_paraphrase_prompts)`

物理意义：

把真实策略控制与普通 prompt 措辞变化分离开。

### Model Identity Effect

定义：

比较以下四类 family-label disagreement 或 major-family distribution distance：

- 相同模型，相同 prompt。
- 同一模型，不同策略。
- 不同模型，同一策略。
- 不同模型，不同策略。

物理意义：

如果 same-strategy cross-model disagreement 更大，说明模型身份/输出风格也是多样性来源；如果 different-strategy same-model disagreement 更大，说明策略 prompt 是更强来源。P4 的目标是定量分解二者，而不是预设 prompt 效应一定大于模型效应。

### Optimization Signal Rate

定义：

在 candidate prompt 评估中：

`signal_rate = count((family_shift_rate > 0 or rho_reduction > 0) and invalid_delta <= invalid_tolerance) / total_candidates_or_updates`

物理意义：

衡量 reward landscape 中是否存在足够多“多样性提升且无效轨迹不恶化”的候选。如果 signal rate 很低，说明指标或 rewriter/search 可能过严。

当前日志中 candidate 诊断位于 `train_step_logs.jsonl` 的顶层 `candidate_behavior_diagnostics` 字段。`scripts/analyze_prove_experiments.py` 会读取：

- `family_shift_rate`
- `rho_reduction`
- `invalid_delta`
- `summary_embedding_shift`

并输出 `optimization_signal_rate`。

## 统计检验

使用简单、稳健的检验：

- 对 question-level diversity delta 做 paired bootstrap confidence interval。
- 对 paired question-level 指标差异做 Wilcoxon signed-rank test。
- 计算 GPT-5.5 method-diversity 分数与策略树 diversity 的 Spearman correlation。
- 如果有多个标注者，计算 Cohen's kappa 或 Krippendorff's alpha。
- 报告 effect size 和置信区间，不只报告 p-value。

推荐 bootstrap：

- 以 question id 为单位有放回重采样。
- 每次 bootstrap 计算 mean delta。
- 取 2.5% 和 97.5% 分位数作为 95% CI。
- 如果 CI 不跨 0，则方向上较稳健。

已实现位置：

- `scripts/analyze_prove_experiments.py`：P2/P3 的 question-level paired bootstrap CI、Wilcoxon 近似检验，以及 P4 model identity check。
- `scripts/analyze_taxonomy_granularity.py`：P6 各粒度指标的 bootstrap CI，以及可选 GPT-5.5/人工分数 Spearman。
- `scripts/run_gpt_blind_validation.py`：P7 GPT-5.5 盲评分数与策略树/文本多样性的 Spearman。
- `scripts/analyze_task_dependence.py`：P8 subject-level intervention effect bootstrap CI。

## 通过标准汇总

| 主张 | 主要证据 | 通过阈值 |
|---|---|---|
| 同 trace 可靠 | repeated judge agreement | major >= 0.85，primary >= 0.70 |
| 不等于表面措辞 | same-strategy paraphrase control | family diversity 低或中等，same-major 高 |
| 对策略干预敏感 | explicit mixed strategies | mixed 相对 same 的 family/major diversity 配对差异为正，homogeneity 下降；有效 trace 口径方向保持 |
| 目标策略遵循诊断 | target exact/same-major hit 与 GPT-5.5 复核 | exact hit 不作硬阈值，重点解释策略可执行性和 judge/taxonomy 吸附 |
| 不等于模型身份 | cross-LLM comparison | 同时报告 prompt effect 与 model identity effect；若模型效应更强，作为跨模型解释风险 |
| 独立盲评有效性 | GPT-5.5 blind ratings | Spearman 正相关，高低组可分 |
| 可优化 | reward sweep | 至少一个非零 reward 设置提升验证集 diversity |
| 不过度约束 | candidate signal 与 early stopping | signal rate 非零，中等设置不应立即停滞 |

## 可能结果的解释

强验证：

- P1 稳定。
- P2 虚假多样性低。
- P3 干预效应强。
- P4 能清楚分解 prompt effect 与 model identity effect，并说明跨模型差异的解释边界。
- P5 中等 reward 提升验证集 diversity。

指标有效但 reward 太强：

- P1-P4 通过。
- P5 失败，表现为 update applied rate 低、candidate shift rate 低，或早停无验证集提升。
- 应调整 reward softness、rewriter 或 candidate evaluation，而不是直接否定指标。

judge/taxonomy 问题：

- P1 失败，或 P7 GPT-5.5 盲评判断与指标明显不一致。
- 先修 judge prompt、taxonomy 粒度或 label 定义，再训练。

任务限制多样性：

- P3 在多方法数据集有效，但在某些 MMLU subject 上弱。
- 应按 subject 报告 reachable diversity，不要强迫天然单一路径的问题多样化。

