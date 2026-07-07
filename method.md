# Method

本文档描述当前仓库实现的方法。项目的核心方法是 **case-aware evolutionary prompt search**，不是严格意义上的 reinforcement learning：系统不训练模型参数，也不做策略梯度更新；reward 只用于候选 prompt 的排序、beam 保留和最终 prompt 选择。

一句话概括：

```text
通过 prompt evolution 诱导多个 agent 形成互补推理路径；
训练时奖励能补足团队盲区的 useful diversity；
推理时默认使用 majority vote，并可选 weighted_vote 来利用少数派正确路径。
```

## 1. 方法目标

多 agent 推理通常会让多个 solver agent 同时回答同一道题，再用投票得到团队答案。问题是：多个 prompt 看起来不同，并不代表多个 agent 真的形成了互补推理。如果它们的 reasoning trace 高度相似，系统只是重复了同一种错误或同一种思路。

本项目要优化的不是“表面多样性”，而是 **有用的互补性**：

- target agent 是否形成了不同但有效的推理路径。
- target agent 是否保持或提升自身正确率。
- 当团队多数投票错误时，是否有 agent 提供了正确少数派路径。
- 候选 prompt 是否增加了团队 answer coverage。
- diversity 是否来自有效、正确的 trace，而不是空泛、无效、重复输出。

## 2. 系统组成

系统包含三类模型角色：

- Solver agents：真正解题，输出 reasoning trace 和最终答案。
- Prompt optimizer：根据窗口案例和诊断信息生成候选 prompt。
- Evaluator：判断 target agent 是否执行了候选 prompt 描述的角色程序。

Optimizer 不直接改写最终系统。候选 prompt 必须先经过 candidate evaluation，再按 reward 进入 per-agent beam search。

## 3. 任务抽象

任务解析集中在 `multi_dataset_diverse_rl/tasks.py`：

```python
@dataclass
class TaskSpec:
    name: str
    parse_gold: Callable[[Any, Optional[str]], str]
    extract_pred: Callable[[Optional[str], Optional[str]], str]
    match_answer: Callable[[str, str], bool]
    format_question: Optional[Callable[[dict], str]] = None
```

`TraceBeamSearchSystem` 通过 `self.task_spec` 完成：

- gold parsing
- prediction extraction
- answer matching

当前支持：

- `mmlu`
- `gsm8k`
- `bbh`
- `auto`

当运行 task-level comparison 并提供 `--answer_format` 时，系统会使用 `multi_dataset_diverse_rl/answer_formats.py`。支持格式包括：

- `option_letter`
- `boolean`
- `yes_no`
- `valid_invalid`
- `numeric`
- `free_text`

## 4. 数据格式

CLI 会把原始记录标准化为：

```json
{"question": "...", "answer": "..."}
```

`--dataset_format legacy` 支持：

- question 字段：`question`, `input`, `query`, `problem`
- answer 字段：`answer`, `output`, `target`, `label`, `response`

`--dataset_format mars` 支持：

- question 字段：`question`, `input`, `query`, `problem`, `prompt`
- answer 字段：`answer`, `target`, `gold`, `gold_answer`, `label`, `output`
- task metadata：`task`, `task_name`, `category`, `subject`, `bbh_task`

如果某条记录无法抽取 question 或 answer，`build_dataset` 会抛出带 record index 的 `ValueError`。

## 5. Rollout

一次 rollout 会让所有 active prompts 同时回答同一道题：

```text
question + active_prompt_i -> trace_i + answer_i
```

每次 rollout 计算：

- `individual_correct`：每个 agent 是否答对。
- `vote_answer` / `vote_correct`：聚合后的团队答案和正确性。
- `any_correct`：是否至少一个 agent 答对。
- `invalid_flags`：trace 是否无效。
- `embedding_diversity`：trace 级语义多样性。
- `mean_embedding_overlap`：trace 平均重叠度。
- vote tie diagnostics：平票、候选答案、计数、tie-break 方法。
- weighted vote diagnostics：可选 diversity-aware aggregation 诊断。

Solver trace 应包含显式最终答案，通常为：

```text
FINAL_ANSWER: <answer>
```

## 6. Trace 有效性

只奖励“不同”是不够的，因为无效、空泛、重复的输出也可能看起来不同。系统使用 rule-based invalid checker 过滤明显坏 trace。

常见 invalid 原因：

- trace 太短。
- 缺少 `FINAL_ANSWER:`。
- token 数过少。
- 无法抽取最终答案。
- bigram 重复比例过高。

无效 trace 不会获得虚假的 diversity bonus。在 embedding overlap 诊断中，涉及 invalid trace 的 pair 会按高重叠处理。

## 7. Trace Diversity

系统用完整 reasoning trace 衡量多样性，而不是只比较 prompt 文本。

流程：

1. 归一化空白。
2. 长 trace 按 `trace_embedding_chunk_words` 分块。
3. 用 `sentence-transformers` 编码。
4. 对 chunk embedding 做平均池化。
5. 计算 agent pair 的 cosine overlap。
6. 得到：

```text
embedding_diversity = 1 - mean_embedding_overlap
```

对 useful diversity，系统只考虑 **valid 且 correct** 的 trace。

## 8. 答案聚合

### 8.1 Majority Vote

默认聚合方式仍然是 majority vote。系统会记录诊断信息：

```json
{
  "vote_answer": "A",
  "vote_tie": true,
  "tie_candidates": ["A", "B"],
  "vote_counts": {"A": 1, "B": 1},
  "tie_break_method": "random"
}
```

Tie-break 策略：

- `first`：保留旧行为，返回最早出现的平票答案。
- `random`：用 `seed + question_hash` 做 deterministic random。
- `abstain`：平票时返回空答案。

### 8.2 Weighted Vote

`--aggregation_mode weighted_vote` 是可选的 diversity-aware aggregation。它保留 majority vote 的全部诊断，同时可以用有效性和独立性权重选择最终答案。

权重形式：

```text
weight_i = reliability_i * validity_i * independence_i
independence_i = min(max(1 - per_agent_overlap_i, 0), 0.5)
score(answer) = sum(weight_i for agents predicting answer)
```

当前 reliability 使用均匀权重；validity 来自 invalid flags；independence 来自 per-agent trace overlap。这样，重复的多数派路径不会天然压过一个有效且独立的少数派正确路径。

日志会同时保留：

- `majority_vote_answer`
- `majority_vote_correct`
- `weighted_vote_answer`
- `weighted_vote_correct`
- `majority_vote_acc`
- `weighted_vote_acc`

## 9. Prompt Evolution

每个 agent 都维护自己的 prompt beam。系统不会每道题都更新 prompt，而是用 `update_every` 积累一个窗口。

窗口中保存：

- question hash
- traces
- answers
- prompts
- rollout metrics
- homogeneous cases
- validity cases

窗口满后，系统选择需要更新的 target agent，并让 optimizer 根据窗口证据生成候选 prompt。

候选 prompt 的证据来源包括：

- high-overlap trace pairs
- mixed window cases
- validity-focused cases
- accuracy-error cases
- previous beam prompts

候选 prompt 必须是可执行的角色程序，通常包含 role name、decision procedure、fallback strategy、accuracy checks、validity checks 和 anti-overlap rule。

## 10. Candidate Evaluation

对某个 target agent，系统在同一个 eval batch 上比较：

```text
baseline_prompts  = current active prompts
candidate_prompts = current active prompts with target agent replaced by candidate prompt
```

这种设计让 candidate evaluation 是 baseline-relative 的，而不是孤立评价一个 prompt。

每个 eval sample 会记录：

- `baseline_vote_correct`
- `candidate_vote_correct`
- `baseline_any_correct`
- `candidate_any_correct`
- `baseline_target_correct`
- `target_agent_correct`
- `target_trace_novelty`
- `target_useful_diversity`
- `rescue`
- `rescue_useful_diversity`

核心定义：

```text
target_useful_diversity =
    target_trace_novelty
  * target_agent_correct
  * target_valid

rescue =
    baseline vote is wrong
    and target agent is correct
```

Candidate eval 支持：

- `random`
- `fixed_pool`
- `stratified`

推荐使用：

```bash
--candidate_eval_strategy fixed_pool
--candidate_eval_batch_size 20
--candidate_eval_pool_size 100
```

## 11. Coverage Useful Diversity Reward

当前主 reward 是 `coverage_useful_diversity`。它奖励的是有用互补性，而不是裸 diversity。`coverage_rescue_diversity` 作为 deprecated alias 保留兼容。

Batch 级聚合指标：

```text
baseline_team_accuracy    = mean(baseline_vote_correct)
candidate_team_accuracy   = mean(candidate_vote_correct)

baseline_oracle_acc       = mean(baseline_any_correct)
candidate_oracle_acc      = mean(candidate_any_correct)
coverage_delta            = candidate_oracle_acc - baseline_oracle_acc

baseline_target_accuracy  = mean(baseline_target_correct)
candidate_target_accuracy = mean(target_agent_correct)

rescue_rate               = mean(rescue)
useful_diversity          = mean(target_useful_diversity)
rescue_useful_diversity   = mean(rescue_useful_diversity)
vote_delta                = candidate_team_accuracy - baseline_team_accuracy
```

Guard：

```text
invalid_guard_passed =
    candidate_invalid_rate <= baseline_invalid_rate + invalid_guard_epsilon

```

Reward：

```text
if not invalid_guard_passed:
    reward = -1.0


else:
    reward =
        candidate_target_accuracy
      + reward_weight_coverage         * coverage_delta
      + reward_weight_useful_diversity * useful_diversity
```

解释：

- `candidate_target_accuracy`：直接奖励被更新 agent 的准确率。
- `coverage_delta`：奖励 candidate 增加 oracle coverage，即使 vote 暂时没有翻转。
- `useful_diversity`：只有 target trace 有效且正确时，新颖性才有价值。
- invalid guard：防止无效 diversity 被选中。
- `rescue_rate` 和 `vote_delta` 只作为诊断日志保留，不进入 reward。
- 本版本移除了 local validity evaluator；候选是否值得保留由 target-agent accuracy、coverage_delta、useful_diversity 和 invalid guard 决定。

推荐运行：

```bash
python -m multi_dataset_diverse_rl.cli \
  --reward_mode coverage_useful_diversity \
  --candidate_eval_strategy fixed_pool \
  --agents 5 \
  --init_mode bank
```

使用 weighted vote：

```bash
python -m multi_dataset_diverse_rl.cli \
  --reward_mode coverage_useful_diversity \
  --aggregation_mode weighted_vote \
  --candidate_eval_strategy fixed_pool \
  --agents 5 \
  --init_mode bank
```

## 12. Beam Selection

候选 prompt 评估完成后，系统按 reward 降序排序，并为该 agent 保留 top `beam_size`。beam top-1 成为新的 active prompt。

`update_logs.jsonl` 会记录每个 candidate：

- reward
- candidate source
- beam rank
- accepted / rejected
- baseline 和 candidate accuracy
- oracle coverage metrics
- rescue metrics
- useful diversity metrics
- invalid guard
- solver reuse statistics

这让 prompt evolution 的每一步都可追踪。

## 13. Validation

在 `coverage_useful_diversity` 下，validation score 为：

```text
0.4 * vote_acc
+ 0.3 * oracle_acc
+ 0.2 * mean_useful_diversity
- 0.2 * mean_invalid_rate
```

对应 metric name：

```text
vote+oracle+useful_div-invalid
```

训练过程中，系统按 validation score 保存 best prompts。训练结束后恢复 best prompts，再做 final test。

## 14. Dataset-Level Metrics

最终评估会输出：

- `vote_acc`：当前 aggregation mode 的准确率。
- `majority_vote_acc`：majority vote 准确率。
- `weighted_vote_acc`：weighted vote 准确率。
- `mean_individual_acc`：所有 agent 预测的平均准确率。
- `best_individual_acc`：最佳单 agent 准确率。
- `oracle_acc`：至少一个 agent 答对的比例。
- `aggregation_gap`：`oracle_acc - vote_acc`。
- `rescue_available_rate`：vote 错但至少一个 agent 对的比例。
- `correct_disagreement_rate`：agent 答案不一致且至少一个 agent 对的比例。
- `mean_useful_diversity`：valid/correct traces 之间的平均多样性。
- `vote_tie_rate`：投票平票比例。
- `mean_invalid_rate`：无效 trace 比例。

这些指标回答三个问题：

1. 团队里是否有人知道正确答案？看 `oracle_acc`。
2. 聚合规则是否利用了这个答案？看 `vote_acc` 和 `aggregation_gap`。
3. 多样性是否有用？看 `mean_useful_diversity` 和 `rescue_available_rate`。

## 15. Task-Level Accuracy Export

task-level export 让 MAD 能按 MARS-style task_id 粒度运行，但不依赖 MARS 仓库。

MAD 自己维护 manifest：

```text
configs/task_level_comparison.yaml
```

每个任务条目包含：

- `task_id`
- `benchmark`
- `task_type`
- `answer_format`
- `train_path`
- `val_path`
- `test_path`

运行：

```bash
python scripts/run_task_level_accuracy.py \
  --manifest configs/task_level_comparison.yaml \
  --benchmarks BBH,MMLU \
  --settings shared_baseline,shared_guarded_beam,bank_guarded_beam \
  --seeds 42 \
  --dataset_format mars \
  --out_root runs_task_level_accuracy
```

输出：

```text
runs_task_level_accuracy/accuracy_results.jsonl
```

每行包含：

- task metadata
- setting 和 seed
- `vote_acc`
- `majority_vote_acc`
- `weighted_vote_acc`
- `mean_individual_acc`
- `best_individual_acc`
- oracle / rescue metrics
- LLM call 和 token 统计

与 MARS 对比时，MAD core 不读取 MARS 仓库；外部脚本只用 `task_id` join：

```bash
python scripts/compare_external_accuracy.py \
  --mars_summary path/to/mars/summary.csv \
  --mad_results runs_task_level_accuracy/accuracy_results.jsonl \
  --out_csv comparison/mars_vs_mad_accuracy.csv \
  --out_md comparison/mars_vs_mad_accuracy.md
```

对比时不要只报告 MAD `vote_acc`。因为 MAD 是多 agent 投票，MARS-style 结果通常是 single-prompt accuracy，所以表格至少应包含：

- MARS Acc
- MAD Vote Acc
- MAD Mean Agent Acc
- MAD Best Agent Acc

## 16. Cost Statistics

每个 run 写入：

```text
llm_calls.jsonl
cost_summary.json
```

统计字段包括：

- `solver_calls`
- `optimizer_calls`
- `evaluator_calls`
- `total_llm_calls`
- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- `estimated_cost`
- `latency_seconds`

这些字段只用于报告和后续分析，不用于 early stop、搜索约束、reward 归一化或 prompt 排序。

## 17. 代码定位

核心文件：

```text
multi_dataset_diverse_rl/config.py
multi_dataset_diverse_rl/cli.py
multi_dataset_diverse_rl/system.py
multi_dataset_diverse_rl/tasks.py
multi_dataset_diverse_rl/answer_formats.py
scripts/run_task_level_accuracy.py
scripts/compute_experiment_metrics.py
scripts/task_level_accuracy_utils.py
```

`TraceBeamSearchSystem` 里的关键方法：

- `compute_rollout_metrics`
- `_weighted_vote_with_diagnostics`
- `_target_trace_novelty`
- `_candidate_reward_coverage_useful_diversity`
- `evaluate_candidate_prompt`
- `update_prompt_with_beam`
- `_summarize_rollout_rows`
- `evaluate_dataset`

## 18. 方法边界

- 系统优化的是 prompt，不训练模型权重。
- Reward 是 candidate ranking signal，不是严格 RL return。
- Trace embedding 是推理相似度近似，不等价于人工策略标签。
- Candidate evaluation 仍有采样方差，需要 fixed pool、stratified sampling 和多 seed。
- `weighted_vote` 是轻量聚合规则，不是 learned verifier。
- task-level comparison 依赖本仓库 manifest；如果 manifest 未覆盖所有目标任务，只能称为 subset comparison。

