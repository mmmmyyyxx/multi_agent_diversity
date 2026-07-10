# Multi-Agent Diversity Prompt Search

## Checkpoint resume

Training runs write `training_checkpoint.json` under each run directory. Pass `--resume_from_checkpoint 1` to resume an incomplete run from the last saved training batch or epoch boundary without deleting existing logs. In task-level experiments, combine it with `--resume_completed 1`: completed run directories are skipped, while incomplete run directories continue from their checkpoint.

```bash
python scripts/run_task_level_accuracy.py \
  --manifest configs/task_level_comparison.yaml \
  --tasks disambiguation_qa,geometric_shapes,ruin_names,sports_understanding \
  --settings shared_baseline,shared_guarded_beam \
  --seeds 42 \
  --dataset_format mars \
  --out_root runs_task_level_bbh_tcs_useful_full \
  --resume_completed 1 \
  --resume_from_checkpoint 1
```

Checkpoint resume is batch/epoch-level. If a process is interrupted in the middle of an in-flight API batch, that small batch may be repeated; recorded solver rollouts are reused when `--candidate_reuse_recorded_rollouts 1` is enabled. If an existing checkpoint was created with incompatible resume-critical settings, resume now fails fast and prints the mismatched fields instead of silently restarting in the same run directory.

本项目是一个多智能体推理实验框架：多个 solver agent 同时回答同一道题，系统记录每个 agent 的 reasoning trace 和最终答案，用多数投票得到团队答案，并用 trace embedding overlap 衡量 agent 之间是否真的形成了不同解题路径。

当前方法更准确地说是 **case-aware evolutionary prompt search**，不是严格意义上的 reinforcement learning。`optimizer_model` 是 prompt-evolution generator model：在默认 Teacher-Critic-Student 架构中，它负责 Teacher、Teacher rewrite、Student、Student JSON retry/repair 等生成侧调用；`evaluator_model` 负责 Critic 和可选 joint trace diversity evaluator。候选 prompt 必须经过小批量评估、reward 排序和 per-agent beam search 后才会被保留。

## 核心能力

- 多 agent 并行 rollout：同题多 prompt、多 trace、多答案。
- TaskSpec 任务抽象：统一 gold parsing、prediction extraction 和 answer matching。
- 支持任务：`mmlu`、`gsm8k`、`bbh`、`auto`。
- 支持数据格式：`legacy` 与 `mars`。
- Trace embedding diversity：用完整 trace 的 embedding overlap 衡量同质化。
- Evolutionary beam search：每个 agent 保留自己的 prompt beam。
- Accuracy-guarded reward：优先保证团队准确率不被多样性目标带崩。
- Candidate evaluation 稳定化：支持随机、固定评估池、分层采样和 repeated evaluation。
- Majority vote diagnostics：记录平票、候选答案、计数和 tie-break 策略。
- 批量实验协议：默认运行两个 baseline 和两个 guarded beam 方法。

## 目录

```text
multi_dataset_diverse_rl/
  cli.py       # 主入口：读取数据、训练、验证、测试
  config.py    # 命令行参数和默认配置
  policy.py    # AgentState 与 prompt beam 状态
  system.py    # TraceBeamSearchSystem 主流程
  tasks.py     # TaskSpec、MMLU/GSM8K/BBH/auto 答案解析与匹配
  utils.py     # 兼容工具、投票、JSONL、指标工具

scripts/
  run_task_level_accuracy.py    # task_id 级 accuracy 实验入口
  run_experiments.py            # 数据集级批量实验入口
  compute_experiment_metrics.py # 多数据集、多 seed 汇总
  compare_external_accuracy.py  # 外部 MARS summary 与 MAD 结果 join
  analyze_experiments.py        # 对已有 run 做汇总与绘图
  plot_experiment_results.py    # 通用实验图表
  experiment_config.py          # 共享 setting / dataset 默认配置
  experiment_io.py              # 共享 CSV/JSON/JSONL IO
  task_level_accuracy_utils.py  # task-level 标准结果行与 schema
  prepare_mmlu_data.py          # MMLU JSONL 准备脚本
  sample_jsonl_splits.py        # 数据采样与 split

tests/
  test_tasks_mmlu.py
  test_tasks_bbh.py
  test_vote.py
  test_reward_guard.py
  test_dataset_format.py
```

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

项目使用 OpenAI-compatible Chat Completions API：

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://your-endpoint/v1"
```

Windows PowerShell：

```powershell
$env:OPENAI_API_KEY="..."
$env:OPENAI_BASE_URL="https://your-endpoint/v1"
```

## 数据格式

主程序读取 JSONL。每行会被标准化成：

```json
{"question": "...", "answer": "..."}
```

也可以直接读取 `Dataset_format/` 下这类带 `question,answer` 表头的 CSV 文件；CLI 会按扩展名自动用 CSV reader 加载。

`--dataset_format legacy` 支持旧字段：

- question: `question`, `input`, `query`, `problem`
- answer: `answer`, `output`, `target`, `label`, `response`

`--dataset_format mars` 支持 MARS-style 字段别名：

- question: `question`, `input`, `query`, `problem`, `prompt`
- answer: `answer`, `target`, `gold`, `gold_answer`, `label`, `output`
- task/subtask: `task`, `task_name`, `category`, `subject`, `bbh_task`

如果某条记录无法抽取 question 或 answer，`build_dataset` 会抛出包含 record index 的 `ValueError`。

示例：

```json
{"prompt": "Evaluate the boolean expression: not false.", "gold": "yes", "task_name": "boolean_expressions"}
```

## 任务类型

通过 `--task_type` 指定答案解析与匹配方式：

- `mmlu`：解析 `A/B/C/D`、`(A)`、`option A` 等选项答案。
- `gsm8k`：兼容 `#### 72`，数值比较时去逗号并归一化 `3.0 -> 3`。
- `bbh`：独立 BBH 解析，不再 fallback 到 GSM8K 的“取最后一个数字”逻辑。
- `auto`：先识别 MMLU/GSM8K，无法确定时按 BBH 风格解析。

BBH 预测抽取优先级：

1. 最后一处 `FINAL_ANSWER: <content>`。
2. 最后一处 `Answer: <content>`。
3. 没有显式标记时取最后一个非空行。

BBH 会归一化 yes/no/true/false、选项 `(A)`、`A.`、`option A`、整数小数等；支持 gold aliases list，例如 `["yes", "true"]`。

## Reward Modes

`--reward_mode guarded_diversity` 是当前默认模式。它先在同一个 eval batch 上计算 baseline prompts 和 candidate prompts 的指标。Reward 使用 target agent accuracy 做 guard 和主准确率项，team vote accuracy 只作为诊断日志保留：

```text
acc_delta = candidate_target_acc - baseline_target_acc
vote_delta = candidate_team_acc - baseline_team_acc
div_delta = candidate_embedding_diversity - baseline_embedding_diversity
invalid_delta = candidate_invalid_rate - baseline_invalid_rate

if candidate_target_acc < baseline_target_acc - accuracy_guard_epsilon:
    reward = -1.0 + acc_delta - reward_weight_invalid_delta * max(0, invalid_delta)
else:
    reward = (
        candidate_target_acc
        + reward_weight_div_delta * div_delta
        - reward_weight_invalid_delta * max(0, invalid_delta)
    )
```

可选模式：

- `guarded_diversity`：默认推荐，准确率约束下优化 trace diversity。
- `accuracy_only`：消融模式，按被更新 target agent 自身准确率选择候选，同时记录 team vote accuracy。
- `coverage_useful_diversity`：奖励 target-agent accuracy、oracle coverage gain 和 useful diversity。

`update_logs.jsonl` 会记录 guard 相关字段：

- `baseline_team_accuracy`
- `candidate_team_accuracy`
- `accuracy_delta`
- `baseline_embedding_diversity`
- `candidate_embedding_diversity`
- `diversity_delta`
- `baseline_invalid_rate`
- `candidate_invalid_rate`
- `invalid_delta`
- `accuracy_guard_passed`

## Phase-Adaptive Reward Schedule

The default reward schedule is `--reward_schedule_mode phase_adaptive`. When agents start from the same shared prompt, early updates use stronger diversity and useful-diversity weights so the optimizer is encouraged to create genuinely different reasoning roles. As prompts diverge, or once enough prompt updates have been accepted, the schedule shifts weight back toward target-agent accuracy and a tighter accuracy guard.

Key controls:

```bash
--reward_schedule_mode phase_adaptive
--reward_diversity_warmup_updates 10
--reward_weight_div_delta_early 0.8
--reward_weight_div_delta_late 0.2
--reward_weight_useful_diversity_early 0.5
--reward_weight_useful_diversity_late 0.25
--reward_weight_target_accuracy_early 0.9
--reward_weight_target_accuracy_late 1.0
--accuracy_guard_epsilon_early 0.03
--accuracy_guard_epsilon_late 0.01
```

`update_logs.jsonl` records the effective weights used for each candidate, including `effective_weight_target_accuracy`, `effective_weight_div_delta`, `effective_weight_coverage`, `effective_weight_useful_diversity`, `reward_phase_progress`, `reward_diversity_need`, and `reward_unique_prompt_ratio`.

Optimizer fallback templates are disabled by default with `--optimizer_fallback_mode none`. If `--optimizer_fallback_mode template` is enabled, fixed fallback candidates are added when the optimizer returns too few valid candidates. Fallback candidates are useful as an engineering safety mechanism, but they can inflate prompt-update success rate and confound attribution to the optimizer. For clean experimental claims, use `--optimizer_fallback_mode none` and report `num_optimizer_candidates`, `num_fallback_candidates`, `num_existing_beam_candidates`, `optimizer_underfilled`, and `top1_candidate_source`.

`update_logs.jsonl` also records optimizer generation diagnostics: `optimizer_raw_response_empty`, `optimizer_json_parse_failed`, `optimizer_raw_candidate_count`, `optimizer_empty_prompt_count`, `optimizer_sanitized_count`, `optimizer_redundant_filtered_count`, `optimizer_schema_filtered_count`, and `optimizer_final_candidate_count`. These fields explain no-op runs where the optimizer produces zero usable candidates.

`accepted` is kept as a deprecated compatibility field meaning "inside the retained beam." Prefer `in_top_beam`, `is_top1`, and `active_prompt_changed` when analyzing actual prompt updates.

No-op evolution early stop is enabled by default. If repeated update attempts produce fewer than `--no_effective_evolution_min_optimizer_candidates` optimizer candidates and no active prompt changes for `--no_effective_evolution_patience` attempts, prompt evolution stops safely and final evaluation still runs. This avoids spending a full training budget on tasks where the optimizer is not producing usable candidates.

## Candidate Evaluation

候选 prompt 不直接采纳，而是在 candidate eval batch 上评估。默认 `candidate_eval_batch_size=20`。

支持策略：

- `random`：从训练集随机抽样，兼容旧行为。
- `fixed_pool`：训练开始时从 val 优先、否则 train 中构建固定评估池，再按 seed/epoch/step 确定性采样。
- `stratified`：根据 `subject/task/category/bbh_task` 尽量均衡抽样。

相关参数：

```bash
--candidate_eval_strategy fixed_pool
--candidate_eval_pool_size 100
--candidate_eval_batch_size 20
--candidate_eval_repeats 1
--candidate_eval_seed_offset 1000
```

Concurrency controls:

```bash
--run_concurrency 2
--optimizer_parent_concurrency 2
--candidate_eval_concurrency 3
--train_rollout_concurrency 8
--eval_solver_call_concurrency 80
```

`run_concurrency` is used by `scripts/run_task_level_accuracy.py` and `scripts/run_experiments.py` to run independent task/setting/seed subprocesses in parallel. `optimizer_parent_concurrency` controls how many beam-parent prompt-generation calls can run at once inside one agent update. `candidate_eval_concurrency`, `train_rollout_concurrency`, and `eval_solver_call_concurrency` control candidate evaluation, training rollout batching, and total solver-call pressure. Increase these gradually; if the API returns quota or transient errors, lower `run_concurrency` first, then `eval_solver_call_concurrency`.

`candidate_eval_repeats > 1` 时，CLI 会构造多个 batch 并合并评估，reward 和 metrics 使用平均值。日志会记录 eval strategy、pool size、batch size 和 repeats。

## Teacher-Critic-Student Optimizer

默认 optimizer 架构是 `--optimizer_architecture teacher_critic_student`。旧的一步式 optimizer 仍保留，可用 `--optimizer_architecture one_shot` 回退。

模型字段含义：

- `agent_model`：solver agents 的解题 rollout。
- `optimizer_model`：prompt-evolution generator model，用于 Teacher、Teacher rewrite、Student、Student JSON retry/repair；在 `one_shot` 架构中也用于直接生成候选 prompt。
- `evaluator_model`：用于 TCS Critic 和可选 joint trace diversity evaluator。

Teacher-Critic-Student 的流程：

- Teacher 不直接写 prompt，而是根据 problem type、answer format、target-agent error type、diversity gap、prompt redundancy、error correlation、invalid-output pattern 和 peer behavior summaries 生成 Socratic guiding question。
- Critic 在 Student 看到问题前审核 Teacher question；如果问题泛化、没有诊断依据、泄漏样本/答案、硬编码任务角色、只追求表面多样性或可能伤害准确率，就拒绝。
- 如果 Critic 拒绝，Teacher 会按反馈重写，最多进行 `teacher_critic_max_rounds` 次 Critic 检测；当前默认是 `3`。
- 如果三次都没有通过阈值，系统会选择 Critic 分数最高的 Teacher question 继续交给 Student，并记录 `teacher_question_forced_best_score=true`。
- Student 根据已批准或最高分的 guiding question 生成候选 prompt。候选仍走原有 reward 和 candidate evaluation，reward 计算不变。

这个 optimizer 不使用预定义任务专属角色，也不默认把 voting failure 作为 Teacher 的主要输入。Voting 是下游聚合方式之一；本步骤优化的是 prompt quality、task alignment、target-agent accuracy 和 useful diversity。

相关日志字段包括 `optimizer_architecture`、`teacher_question`、`teacher_question_approved`、`teacher_question_score`、`teacher_question_forced_best_score`、`teacher_question_forced_best_round`、`teacher_question_forced_below_threshold`、`teacher_critic_rounds`、`teacher_rewrite_count`、`student_candidate_count_raw`、`student_candidate_count_final`、`student_candidate_filtered_count`、`student_all_candidates_filtered`、`diversity_contribution`、`error_correlation_reduction`、`task_alignment_rule` 和 `peer_redundancy_avoidance`。

### Student Failure Diagnostics

When `teacher_question_approved=true` but no optimizer candidates are produced, `update_logs.jsonl` records `student_failure_stage` and related fields to distinguish raw empty output, JSON parse failure, missing `candidates`, wrong candidate type, empty candidate list, refusal/explanation text, schema filtering, redundant filtering, and mixed filtering.

Failure stages include `raw_empty`, `json_parse_failed`, `missing_candidates_key`, `candidates_not_list`, `empty_candidates_list`, `refusal_or_explanation`, `all_candidates_filtered_schema`, `all_candidates_filtered_redundant`, and `all_candidates_filtered_mixed`.

#### JSON Retry and Repair

Student parse failures or empty responses may happen when the model outputs truncated, malformed, or blank JSON. The system retries Student with a stricter JSON-only instruction up to `student_json_max_retries` times; the current default is `5`. If retry still fails on non-empty malformed JSON, it can call a JSON repair utility that only repairs syntax of already generated Student content. This is not a prompt fallback and does not generate template prompts.

Logs include `student_json_retry_attempted`, `student_json_retry_succeeded`, `student_json_repair_attempted`, `student_json_repair_succeeded`, and `student_json_repair_failure_reason`. Recovered parse failures are reported separately by `scripts/analyze_student_failures.py` and are not counted as final Student failures.

Summarize failures with:

```bash
python scripts/analyze_student_failures.py <run_dir>
```

## Majority Vote Tie-Break

多数投票现在会记录诊断信息：

```json
{
  "vote_answer": "A",
  "vote_tie": true,
  "tie_candidates": ["A", "B"],
  "vote_counts": {"A": 1, "B": 1},
  "tie_break_method": "random"
}
```

`--vote_tie_break` 支持：

- `first`：旧行为，返回最早出现的平票答案。
- `random`：默认，用 `seed + question_hash` 做 deterministic random，避免 agent 顺序偏置。
- `abstain`：平票时返回空答案。

`evaluate_dataset` prediction 日志和 `train_step_logs.jsonl` 都会写入 tie 信息；汇总脚本会输出 `vote_tie_rate`。

## 单次运行

MMLU：

```bash
python -m multi_dataset_diverse_rl.cli \
  --task_type mmlu \
  --dataset_format legacy \
  --train_path mmlu_train.jsonl \
  --val_path mmlu_val.jsonl \
  --test_path mmlu_test.jsonl \
  --out_dir runs_trace_beam/mmlu/shared_guarded_seed42 \
  --agents 5 \
  --init_mode shared \
  --epochs 3 \
  --reward_mode guarded_diversity \
  --candidate_eval_strategy fixed_pool
```

BBH：

```bash
python -m multi_dataset_diverse_rl.cli \
  --task_type bbh \
  --dataset_format mars \
  --train_path bbh_train.jsonl \
  --val_path bbh_val.jsonl \
  --test_path bbh_test.jsonl \
  --out_dir runs_trace_beam/bbh/shared_guarded_seed42 \
  --agents 5 \
  --init_mode shared \
  --epochs 3
```

只跑 baseline：

```bash
python -m multi_dataset_diverse_rl.cli \
  --task_type mmlu \
  --test_path mmlu_test.jsonl \
  --out_dir runs_trace_beam/mmlu/shared_baseline_seed42 \
  --agents 5 \
  --init_mode shared \
  --baseline_only 1
```

## 推荐批量实验

推荐论文式实验命令：

```bash
python scripts/run_experiments.py --datasets mmlu,bbh --seeds 42,43,44
```

默认 settings：

- `shared_baseline`
- `bank_baseline`
- `shared_guarded_beam`
- `bank_guarded_beam`

这些默认实验设置集中定义在 `scripts/experiment_config.py`，`run_experiments.py`、`analyze_experiments.py` 和测试都复用同一份配置。

默认会对所有 setting 使用 `--seeds` 中的每个 seed；如果想省成本，可以加 `--seed_baselines 0` 让 baseline 只跑第一个 seed。

输出目录命名：

```text
runs_trace_beam/
  mmlu/
    shared_baseline_seed42/
    shared_guarded_beam_seed42/
    bank_guarded_beam_seed42/
  bbh/
    shared_baseline_seed42/
    shared_guarded_beam_seed42/
```

默认路径：

- MMLU: `mmlu_train.jsonl`, `mmlu_val.jsonl`, `mmlu_test.jsonl`
- BBH: `bbh_train.jsonl`, `bbh_val.jsonl`, `bbh_test.jsonl`

也可以覆盖：

```bash
python scripts/run_experiments.py \
  --datasets mmlu,bbh \
  --seeds 42,43 \
  --mmlu_train_path data/mmlu_train.jsonl \
  --mmlu_val_path data/mmlu_val.jsonl \
  --mmlu_test_path data/mmlu_test.jsonl \
  --bbh_train_path data/bbh_train.jsonl \
  --bbh_val_path data/bbh_val.jsonl \
  --bbh_test_path data/bbh_test.jsonl
```

选择部分 settings：

```bash
python scripts/run_experiments.py \
  --datasets mmlu \
  --run_settings shared_baseline,shared_guarded_beam \
  --seeds 42
```

## 汇总实验

```bash
python scripts/compute_experiment_metrics.py \
  --runs_root runs_trace_beam \
  --out_csv runs_trace_beam/experiment_metrics.csv \
  --out_md runs_trace_beam/experiment_metrics.md \
  --out_group_csv runs_trace_beam/experiment_metrics_grouped.csv
```

汇总会递归读取多 dataset 子目录，并按 `dataset/setting` 输出 seed mean/std。核心列包括：

- `latest_test_vote_acc`
- `latest_test_embedding_diversity`
- `latest_test_invalid_rate`
- `vote_tie_rate`
- `solver_calls`
- `solver_reuse_hit_rate`

如果提供 MARS baseline：

```bash
python scripts/compute_experiment_metrics.py \
  --runs_root runs_trace_beam \
  --mars_result_path mars_results.jsonl
```

会额外输出：

- `vs_mars_delta_acc`
- `vs_mars_delta_diversity`

## 输出文件

每个 run 的 `out_dir` 主要包含：

```text
run_meta.json
history.json
prompt_history.json
update_logs.jsonl
train_step_logs.jsonl
train_trace_history.jsonl
test_trace_history.jsonl
val_epochN_predictions.jsonl
test_final_predictions.jsonl
last_state.json
best_state.json
selected_state.json
best_prompts.json
```

重点指标：

- `vote_acc`：majority vote accuracy。
- `vote_tie_rate`：平票比例，用于诊断 vote 聚合稳定性。
- `mean_embedding_diversity`：平均 trace embedding diversity。
- `mean_embedding_overlap`：平均 trace embedding overlap。
- `mean_invalid_rate`：无效 trace 比例。
- `num_optimizer_candidates` / `num_fallback_candidates` / `num_existing_beam_candidates`：候选池来源统计。
- `in_top_beam` / `is_top1` / `active_prompt_changed`：区分进入 beam、成为 top-1、以及 active prompt 是否真的变化。
- `no_effective_evolution_stopped`：连续无效 prompt evolution 是否触发提前停止。
- `reward`：候选 prompt 评估得分。
- `solver_reuse_hit_rate`：candidate evaluation 复用历史 rollout 的比例。

## 测试

```bash
python -m pytest
```

## Task-level accuracy comparison

MARS and multi_agent_diversity are run separately. This repository does not depend on the MARS codebase, MARS configs, or MARS prompts. For later comparison, MAD can run at the same `task_id` granularity and export a standardized `accuracy_results.jsonl`; an external script can then join that file with a separately produced MARS `summary.csv` using `task_id`.

Task-level runs are indexed by this repository-owned manifest:

```text
configs/task_level_comparison.yaml
```

The bundled manifest currently covers the local comparison subset available in `Dataset_format/`: 6 BBH tasks and 6 MMLU subjects. If your MARS table contains a different or larger task set, add matching entries before calling it a full comparison; otherwise report it as a subset comparison.

The current manifest reuses the same CSV as `train_path`, `val_path`, and `test_path` for each task. This is a paper-compatible setting for matching task-level MARS-style runs, not a strict no-leakage split. Exported rows include `split_protocol=paper_compatible_reused_file` and `leakage_warning=true` when this is the case.

Run MAD task-level accuracy export:

```bash
python scripts/run_task_level_accuracy.py \
  --manifest configs/task_level_comparison.yaml \
  --benchmarks BBH,MMLU \
  --settings shared_baseline,shared_guarded_beam,bank_guarded_beam \
  --seeds 42 \
  --dataset_format mars \
  --out_root runs_task_level_accuracy
```

The main MAD comparison score is `vote_acc`, because MAD uses multi-agent majority voting. The export also includes `mean_individual_acc` and `best_individual_acc` so single-prompt methods can be compared more carefully.

Recommended paper table columns are `MARS Acc`, `MAD Vote Acc`, `MAD Mean Agent Acc`, and `MAD Best Agent Acc`. Avoid presenting only MAD vote accuracy when comparing against a single-prompt MARS result.

Each `accuracy_results.jsonl` row includes:

```text
task_id, benchmark, method_id, setting, seed, dataset_format,
split_protocol, leakage_warning, num_test_samples,
vote_acc, mean_individual_acc, best_individual_acc,
solver_calls, optimizer_calls, evaluator_calls, total_llm_calls,
prompt_tokens, completion_tokens, total_tokens, estimated_cost, latency_seconds
```

Answer-format parsing is optional and only activates when `--answer_format` is provided. Supported formats are `option_letter`, `boolean`, `yes_no`, `valid_invalid`, `numeric`, and `free_text`. If `--answer_format` is empty, the existing `task_type` parser is used.

Cost statistics are reported for transparency only. They are not used to constrain search, stop training, rank candidates, or normalize accuracy. There is no cost-matched mode and no call budget limit in this protocol.

External comparison command:

```bash
python scripts/compare_external_accuracy.py \
  --mars_summary path/to/mars/summary.csv \
  --mad_results runs_task_level_accuracy/accuracy_results.jsonl \
  --out_csv comparison/mars_vs_mad_accuracy.csv \
  --out_md comparison/mars_vs_mad_accuracy.md
```

This helper reads only the two user-provided files. It does not assume a MARS repository layout.
It accepts common MARS summary aliases such as `task_id`/`task`/`task_name`/`dataset`, `accuracy`/`acc`, and `method_id`/`method`/`model`.

## Coverage Useful Diversity Reward

`coverage_useful_diversity` is an optional reward mode for making prompt evolution prefer useful diversity instead of raw disagreement. Its reward keeps only four active components: target-agent accuracy, oracle coverage gain, useful diversity, and an invalid-output guard. `rescue_rate` is still logged as a diagnostic, but it is not used in the reward.

Candidate evaluation compares the current active prompts against the candidate-replaced prompts on the same eval batch. It logs `baseline_oracle_acc`, `candidate_oracle_acc`, `coverage_delta`, `rescue_rate`, `useful_diversity`, `rescue_useful_diversity`, `baseline_target_accuracy`, `candidate_target_accuracy`, and `invalid_guard_passed`.

Dataset evaluation now also reports:

- `oracle_acc`: at least one agent is correct.
- `aggregation_gap`: `oracle_acc - vote_acc`.
- `rescue_available_rate`: vote is wrong but at least one agent is correct.
- `correct_disagreement_rate`: agents disagree and at least one answer is correct.
- `mean_useful_diversity`: trace diversity among valid, correct agents.

Run example:

```bash
python -m multi_dataset_diverse_rl.cli \
  --reward_mode coverage_useful_diversity \
  --candidate_eval_strategy fixed_pool \
  --agents 5 \
  --init_mode bank
```

Inference still defaults to majority voting. To try diversity-aware aggregation, use:

```bash
python -m multi_dataset_diverse_rl.cli \
  --reward_mode coverage_useful_diversity \
  --aggregation_mode weighted_vote
```

`weighted_vote` keeps majority diagnostics in the logs while selecting the final `vote_answer` by validity and trace-independence weights. It is intended to expose minority correct paths; `majority_vote_acc` and `weighted_vote_acc` are both exported for comparison.

当前测试覆盖：

- MMLU 旧解析兼容。
- BBH 独立解析，不走 GSM8K 数字 fallback。
- legacy/mars dataset format。
- majority vote tie diagnostics。
- guarded reward accuracy penalty。

## 设计边界

- 本项目依赖真实 LLM API，完整实验有成本。
- Diversity metric 当前以 trace embedding 为主，不能等同于人工定义的策略族差异。
- Evaluator 只用于 local role execution，仍可能有判断噪声。
- Candidate evaluation 仍是抽样估计，因此推荐使用 `fixed_pool` 或 `stratified` 并设置多个 seed。
- 方法是 case-aware evolutionary prompt search，不是严格 RL；reward 只用于候选 prompt 排序和 beam selection。
