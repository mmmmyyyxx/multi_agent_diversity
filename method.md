# Method: Multi-Agent Diversity Prompt Search

## 0. 先读这一页

本项目训练的是 **prompt**，不是模型权重。它让多个 solver agent 用不同 prompt 解同一道题，通过候选 prompt 的小批量评估和每个 agent 自己的 beam search，逐步寻找既能保持个人正确率、又能补足团队盲区的提示词组合。

这是一种 **case-aware evolutionary prompt search**，不是严格的 reinforcement learning：

- 不更新 LLM 权重，也没有 policy-gradient。
- reward 只用于给候选 prompt 排序。
- 主要最终分数是 `vote_acc`，即多 agent 多数投票准确率。
- 同时报告 `mean_individual_acc`、`best_individual_acc`、`oracle_acc`，避免把团队投票和单 agent 能力混为一谈。

一句话目标：

```text
用 prompt 演化诱导互补推理路径；
训练时奖励能补足团队盲区的 useful diversity；
评估时默认使用多数投票，并报告少数正确路径是否被浪费。
```

## 1. 整体流程

```text
任务数据
  -> 多个 solver agent rollout
  -> 窗口诊断：错误、覆盖缺口、无效输出、useful diversity
  -> 选择 1-2 个最值得更新的 agent
  -> 生成候选 prompt（默认 Teacher-Critic-Student）
  -> 与当前团队做同批 candidate evaluation
  -> 每个 agent 保留自己的 top-k prompt beam
  -> validation 选择 best_prompts
  -> 恢复 best_prompts 做最终 test，并导出标准结果
```

核心类是 `multi_dataset_diverse_rl.system.TraceBeamSearchSystem`；单任务入口是 `python -m multi_dataset_diverse_rl.cli`；按 `task_id` 批量运行的入口是 `scripts/run_task_level_accuracy.py`。

## 2. 角色与术语

历史 CLI 字段名保留了 `agent_model`、`optimizer_model`、`evaluator_model`，但当前职责如下：

| 字段 | 当前职责 |
| --- | --- |
| `agent_model` | solver rollout 模型。每个 agent 用它回答 train、validation、test 样本。 |
| `optimizer_model` | prompt-evolution 的生成模型。默认 TCS 中用于 Teacher、Teacher rewrite、Student、Student JSON retry/repair。 |
| `evaluator_model` | 审核模型。用于 TCS Critic，以及可选的联合 trace-diversity 评估。 |
| `embedding_model` | 本地 sentence-transformer，用于 trace embedding 多样性诊断。 |

常用术语：

- **active prompt**：某个 agent 当前实际用于 rollout 的 beam top-1 prompt。
- **prompt beam**：该 agent 保留的 top-`beam_size` 候选 prompt 集合；默认 `beam_size=3`。
- **oracle_acc**：只要任一 agent 答对，该样本就算正确。它衡量团队是否已经产生正确路径。
- **aggregation_gap**：`oracle_acc - vote_acc`。越高说明正确的少数派路径越多，但尚未被最终聚合充分利用。
- **useful diversity**：来自有效且正确 reasoning trace 的新颖性；不是 prompt 文本表面不同，也不是单纯 trace 差异。
- **invalid trace**：过短、缺少 `FINAL_ANSWER:`、无法提取答案或高度重复等规则性无效输出。

`run_meta.json` 中的 `model_role_map` 会保存这些模型角色映射。

## 3. 任务、数据与外部比较

### 任务解析

任务逻辑集中在 `multi_dataset_diverse_rl/tasks.py` 的 `TaskSpec`。支持：

- `mmlu`
- `bbh`
- `gsm8k`
- `auto`

在 task manifest 或 CLI 中给出 `--answer_format` 时，答案解析转由 `multi_dataset_diverse_rl/answer_formats.py` 统一处理。支持 `option_letter`、`boolean`、`yes_no`、`valid_invalid`、`numeric`、`free_text`。

### 数据格式

- `legacy`：兼容 `question/input/query/problem` 与 `answer/output/target/label/response` 等旧字段。
- `mars`：兼容 MARS 风格字段别名，如 `prompt`、`gold`、`task_name`、`subject`。它只是数据字段兼容层，**不依赖本地 MARS 仓库**。

### task-level manifest

`configs/task_level_comparison.yaml` 是本仓库自己的轻量实验索引。它目前覆盖 6 个 BBH 和 6 个 MMLU 子任务；按 `task_id` 输出结果，之后可由外部脚本与任意给定的 MARS `summary.csv` 进行 join。

manifest 中同一任务的 train/val/test 目前可能复用同一个 CSV。这是便于贴近 paper-mode 口径的 **paper-compatible setting**，不是严格无泄漏的 split 评测；正式报告必须如实说明。

## 4. 一次训练更新如何发生

每个训练样本先做一次完整团队 rollout：

```text
question + active_prompt_i -> trace_i + answer_i
```

系统记录每个 agent 的答案、正确性、trace 有效性、投票结果、oracle coverage、平票信息和 trace 多样性。每经过 `update_every` 个训练 step（默认 10），执行一次更新：

1. 从最近窗口生成 reward-oriented diagnosis。
2. 选择一到两个有正向更新压力的 agent。
3. 为每个被选中的 agent 生成候选 prompt。
4. 在同一个 candidate-eval batch 上比较当前团队与“仅替换目标 agent prompt 后”的团队。
5. 按 reward 排序候选和已有 beam prompt，保留 top-`beam_size`。
6. beam top-1 成为新的 active prompt。

候选评估是相对基线的：

```text
baseline_prompts  = 当前 active prompts
candidate_prompts = 用 candidate 替换 target agent 后的 active prompts
```

因此某个 prompt 不是独立地“好”，而是要在当前团队中能带来更高的 reward。

## 5. 更新谁：窗口诊断与 agent 选择

当前默认路径不再只按 overlap pressure 选择 agent。`_window_update_diagnosis(...)` 会聚合最近窗口中的：

- 每个 agent 自身错误数。
- 团队投票错误时该 agent 的错误数。
- 无效输出率。
- oracle coverage 缺口：团队里有人正确但该 agent 没有给出正确路径。
- useful-diversity 缺口。
- trace overlap、同质案例等诊断信息。

`select_reward_agents_for_update(...)` 对 agent 的更新压力为：

```text
score_i =
    3.0 * per_agent_error_count[i]
  + 2.0 * per_agent_team_wrong_error_count[i]
  + 2.0 * per_agent_invalid_rate[i]
  + 1.5 * per_agent_coverage_gap_count[i]
  + 1.0 * per_agent_useful_diversity_deficit[i]
```

只选择正分 agent，随机打乱后用于处理并列，并返回排名前 1 或 2 个。Overlap 仍会被记录并提供给 prompt generation，但不再是默认选择器的主导信号。

## 6. Prompt evolution：默认 Teacher-Critic-Student

默认配置：

```bash
--optimizer_architecture teacher_critic_student
```

旧的 `one_shot` 架构仍可用于消融或兼容实验。

TCS 是审核驱动的修正闭环，而不是 Teacher 与 Critic 自由讨论：

1. **Teacher** 从抽象窗口诊断生成一个 Socratic guiding question，不直接写候选 prompt。
2. **Critic** 在 Student 看到该问题前审核它，检查是否泛泛、无诊断依据、泄漏答案、硬编码任务角色、只追求表面差异或可能伤害准确率。
3. 若 Critic 拒绝，Teacher 接收 feedback 重写，再审核；最多 `teacher_critic_max_rounds` 轮，默认 3。
4. 若所有轮次都未通过阈值，系统选择 Critic 得分最高的 Teacher question 继续，并在日志标记 `teacher_question_forced_best_score=true`。
5. **Student** 根据该 guiding question 生成候选 prompt；候选仍必须通过 schema、去重、candidate evaluation 和 beam selection。

这个流程不使用预定义任务专属角色。它的目标是 prompt quality、任务对齐、目标 agent 正确率和 useful diversity；voting failure 不是 Teacher 的默认唯一目标。

真正由 TCS optimizer 生成的 candidate 必须保存完整 provenance：非空 `teacher_question`、至少一次 `teacher_critic_rounds`，以及正数的 `student_candidate_count_raw/final`。`teacher_question_forced_best_score=true` 是合法路径：Critic 没有通过阈值，但最高分的 Teacher 问题仍交给 Student；此时 `teacher_question_approved=false` 如实表示 Critic 未通过。`existing_beam` candidate 不需要这些字段。违反 provenance 不变量的 TCS optimizer candidate 会在进入 candidate evaluation 前直接报错。

`llm_calls.jsonl` 对每次调用记录标准 `llm_call_stage`：`teacher`、`critic`、`teacher_rewrite`、`student`、`student_json_retry`、`student_json_repair`、`one_shot_optimizer`、`solver`；同时记录 agent、parent、Critic round、模型角色与是否为空响应。可用：

```powershell
python scripts/audit_tcs_run.py <run_dir>
```

审计脚本不修改 run 文件，会检查 TCS provenance、调用阶段证据及 candidate delta 一致性；existing-beam candidate 会被排除在 TCS provenance 校验外。

每个 TCS parent 都有确定性的 `tcs_call_group_id`，贯穿 Teacher、Critic、rewrite、Student、JSON retry/repair 及其候选。审计按 group 而不是按整次 run 汇总阶段：只有产生 candidate 的 group 才必须具备完整成功调用证据。`teacher_question_approved=true, forced_best=false` 与 `approved=false, forced_best=true` 是仅有的合法终态。

Candidate evaluation 有两种执行模式。`legacy` 保留历史调用路径；`factorized_cached` 先取得当前 batch 的固定 peer rollout 和每个唯一 target prompt rollout，再调用相同的 candidate 指标代码重组成当前团队指标。复用的是 `agent_id + solver settings + question hash + prompt hash` 的逐题 rollout，绝不复用旧 batch 的 accuracy、coverage、diversity 或 reward 聚合值。重复 prompt 仅共享 rollout，仍保留自己的 candidate/parent/TCS provenance。当前不支持跨 agent rollout reuse。single-flight 使同一个 cache key 的并发 miss 只触发一次 solver 调用。

正式 BBH pilot 使用 `configs/task_level_comparison_strict_bbh_seed42.yaml`。该 manifest 将每个任务映射到独立的 `opt.csv`、`val.csv` 与 `test.csv`，因此 task-level export 必须标记为 `split_protocol=task_manifest_split` 与 `leakage_warning=false`。默认 manifest 保留给 paper-compatible 数据复用对照，不应用于 strict split 结论。

### Student JSON 稳定性

Student 默认输出紧凑 JSON（`--student_candidate_schema_mode compact`）。默认启用：

```text
student_json_retry_on_parse_fail = True
student_json_max_retries = 5
student_json_repair_enabled = True
```

处理顺序：

```text
Student 初次生成
  -> 空输出或 JSON 解析失败时，用更严格 JSON-only 指令重试
  -> 非空但仍是坏 JSON 时，可调用 JSON repair 修复已有文本语法
  -> 仅接收最终可解析、满足 schema 的 candidate 对象
```

JSON repair 不创造新的 prompt idea，也不是 template fallback。最终是否真正失败应看 `student_candidate_count_final == 0`，不要只看一个中间的 `json_parse_failed` 标记。

## 7. 三种 reward mode

当前只有以下三种 mode：

| mode | 用途 | reward 核心 |
| --- | --- | --- |
| `guarded_diversity` | 默认模式 | 保护目标 agent 准确率，再奖励 trace diversity 增量并惩罚无效输出。 |
| `coverage_useful_diversity` | 用于验证“有用多样性”的推荐模式 | 目标 agent 准确率 + oracle coverage 增量 + useful diversity，并加 invalid guard。 |
| `accuracy_only` | 消融 | 仅按更新后目标 agent 自身准确率排序，团队投票仅记录。 |

### `guarded_diversity`

```text
acc_delta     = candidate_target_acc - baseline_target_acc
div_delta     = candidate_embedding_diversity - baseline_embedding_diversity
invalid_delta = candidate_invalid_rate - baseline_invalid_rate

if candidate_target_acc < baseline_target_acc - effective_accuracy_guard_epsilon:
    reward = -1.0 + acc_delta - effective_invalid_weight * max(0, invalid_delta)
else:
    reward = effective_target_accuracy_weight * candidate_target_acc
           + effective_div_delta_weight * div_delta
           - effective_invalid_weight * max(0, invalid_delta)
```

`candidate_team_accuracy` 与 `vote_delta` 会记录在诊断中，但不构成该 reward 的基础。

### `coverage_useful_diversity`

```text
coverage_delta = candidate_oracle_acc - baseline_oracle_acc

if candidate_invalid_rate > baseline_invalid_rate + invalid_guard_epsilon:
    reward = -1.0
else:
    reward = effective_target_accuracy_weight * candidate_target_accuracy
           + effective_coverage_weight * coverage_delta
           + effective_useful_diversity_weight * useful_diversity
```

此模式中真正生效的组成是：`candidate_target_accuracy`、`coverage_delta`、`useful_diversity` 和 `invalid_guard`。`rescue_rate`、`rescue_useful_diversity`、`candidate_team_accuracy` 和 `vote_delta` 都只是日志诊断，不直接被优化。

所有 candidate 的 baseline-relative delta 由同一函数构造，避免 reward 与日志口径漂移：

```text
accuracy_delta  = candidate_target_accuracy - baseline_target_accuracy
diversity_delta = candidate_embedding_diversity - baseline_embedding_diversity
invalid_delta   = candidate_invalid_rate - baseline_invalid_rate
vote_delta      = candidate_team_accuracy - baseline_team_accuracy
coverage_delta  = candidate_oracle_acc - baseline_oracle_acc
```

`net_coverage_delta` 必须等于 `coverage_delta`（允许浮点误差）。

### `accuracy_only`

`accuracy_only` 的 `reward` 是 `target_agent_accuracy`，不是 team vote accuracy。它适合作为“没有多样性奖励”时的对照。

### 阶段自适应调度

默认 `--reward_schedule_mode phase_adaptive`。在共享 prompt 的早期，较重视 diversity/useful-diversity；随着 prompt 分化和有效更新积累，权重逐步回到目标 agent 正确率并收紧 accuracy guard。每个 candidate 的有效权重会写入 `update_logs.jsonl`，包括 `effective_weight_target_accuracy`、`effective_weight_coverage`、`effective_weight_useful_diversity` 与 `effective_accuracy_guard_epsilon`。

### Candidate selection mode：scalar 或 Oracle Pareto

`--candidate_selection_mode scalar_reward` 是默认且完全保留的旧行为：所有 candidate 按现有 scalar reward 排序，取 top-`beam_size`。

`--candidate_selection_mode oracle_pareto` 不改 reward 的计算，只改候选保留方式。它先对同一 candidate-eval batch 的逐样本 oracle correctness 计算：

```text
coverage_gain: baseline 未覆盖、candidate team 覆盖
coverage_loss: baseline 已覆盖、candidate team 不再覆盖
net_coverage_delta = coverage_gain_rate - coverage_loss_rate
```

candidate 必须同时通过 target-agent accuracy guard 和 invalid-rate guard，才可进入 Pareto front。对可行 candidate 使用三个目标：最大化 `coverage_gain_rate`、最小化 `coverage_loss_rate`、最大化 `candidate_target_accuracy`。前沿按 non-dominated sorting 填充 beam；最后放不下的 front 用 crowding distance 裁剪。`useful_diversity` 只用于确定性 tie-break。

Vote accuracy、vote delta、majority flip、embedding diversity、rescue rate、Student 自述效果和 scalar reward 均不参与 Pareto objective 或 active prompt 排序。scalar reward 仍完整记录，便于和 `scalar_reward` 做消融。

## 8. Beam search、validation 与最终 prompt

每个 agent 维护独立 prompt beam：

1. 现有 beam 中的每个 prompt 都作为 parent。
2. 每个 parent 生成 `num_candidates_per_parent` 个新候选。
3. 现有 beam prompt 也留在候选池中，防止较差更新覆盖已有好 prompt。
4. 所有候选都在 candidate-eval batch 上评估。
5. reward 前 `beam_size` 名组成新 beam，top-1 变成 active prompt。

这解释了两个常见现象：

- 新候选被生成，不代表它一定成为 active prompt；已有 beam 项可能在该评估批次上得分更高。
- `prompt_history.json` 是训练过程的 beam 轨迹；`best_prompts.json` 是 validation 选择并为最终测试恢复的 prompt。最终 test 以 `best_prompts.json` 为准。

在 `oracle_pareto` 模式中，retained beam 先由 Pareto front 决定；然后才从 retained set 按 `net_coverage_delta`、coverage loss、coverage gain、target accuracy、useful diversity、Pareto rank、candidate id 的固定词典序排列，因此 `beam[0]` 是确定性的 active prompt。当前 active prompt 始终在候选池中作为保底项。

`--best_state_selection_mode oracle_first` 可让 validation 选择 best state 时按以下顺序决胜：最大 `oracle_acc`、最大 `mean_individual_acc`、最小 `mean_invalid_rate`、最大 `mean_useful_diversity`、更早 epoch。它不使用 vote accuracy、aggregation gap 或复合 scalar validation score；默认 `existing` 保持旧选择行为。

## 9. 聚合和主要指标

默认聚合为多数投票：`--aggregation_mode majority`。平票默认使用 `--vote_tie_break random`，随机性由 `seed + question_hash` 确定，因此可复现；可改为 `first` 或 `abstain`。

`weighted_vote` 是可选轻量规则，用 trace 有效性和独立性降低冗余答案的影响，不是学习得到的 verifier。

`verifier_select` 目前尚未实现实际 verifier；选择它会记录 fallback 并退回 majority，不能作为独立方法报告。

最终结果中最值得一起查看的字段：

| 指标 | 解读 |
| --- | --- |
| `vote_acc` | 主 MAD 指标：最终团队多数投票是否正确。 |
| `mean_individual_acc` | 所有 agent 自身准确率的均值。 |
| `best_individual_acc` | 最强单 agent 的准确率。 |
| `oracle_acc` | 团队是否至少有一个正确路径。 |
| `aggregation_gap` | 正确少数路径尚未转化为最终投票的空间。 |
| `mean_useful_diversity` | 多样性是否来自有效、正确的 reasoning trace。 |
| `mean_invalid_rate` | 无效输出比例。 |

对外比较时不要只报告 `vote_acc`；至少并列 MARS Acc、MAD Vote Acc、MAD Mean Agent Acc、MAD Best Agent Acc，并注明前者可能是单 prompt、后者是多 agent 投票。

## 10. 输出文件：先看什么

每个 run 目录通常包含：

| 文件 | 用途 |
| --- | --- |
| `run_meta.json` | 完整配置、数据元信息与模型角色。 |
| `history.json` | 每个 epoch 的 validation/test 汇总。 |
| `best_prompts.json` | validation 选出的最终测试 prompt。 |
| `prompt_history.json` | 每个 agent 的 beam/active prompt 演化轨迹。 |
| `update_logs.jsonl` | 每次候选评估与 beam 更新：reward、来源、是否进入 beam、是否成为 top-1。 |
| `train_step_logs.jsonl` | 逐训练 step 的 rollout 指标与诊断。 |
| `solver_rollout_records.jsonl` | 可复用的 solver rollout 记录。 |
| `llm_calls.jsonl` | 单次 LLM 调用、token、延迟与错误。 |
| `cost_summary.json` | LLM 调用与 token 汇总，只用于报告。 |
| `training_checkpoint.json` | 未完成训练的断点；正常完成时会清除。 |

分析更新效果时优先看 `update_logs.jsonl`：

- `active_prompt_changed`：这次 beam 更新是否真的改变了 active prompt。
- `is_top1` / `in_top_beam`：候选是否成为当前 top-1 / 进入保留 beam。
- `top1_candidate_source`：最终 top-1 来自 optimizer、已有 beam 还是 fallback。
- `student_candidate_count_final`：Student 最终产生的可用候选数。
- `teacher_question_forced_best_score`：三轮 Critic 未过阈值时是否采用最高分 Teacher question。
- `coverage_gain_count/rate`、`coverage_loss_count/rate`、`net_coverage_delta`：candidate 对 oracle coverage 的逐样本变化。
- `pareto_feasible`、`pareto_rank`、`pareto_crowding_distance`、`pareto_selected`：Oracle Pareto 的可行性、front、裁剪和最终保留结果；scalar 模式下这些字段为 `null`。

成本字段包括 `solver_calls`、`optimizer_calls`、`evaluator_calls`、`total_llm_calls`、token、`estimated_cost` 和 `latency_seconds`。它们只用于报告，**不会**用于预算限制、早停、排序或归一化训练。

## 11. 断点续跑

训练会在 step、批处理和 epoch 边界写入 `training_checkpoint.json`。使用：

```powershell
python scripts/run_task_level_accuracy.py `
  --manifest configs/task_level_comparison.yaml `
  --tasks disambiguation_qa,geometric_shapes,ruin_names,sports_understanding `
  --settings shared_baseline,shared_guarded_beam `
  --seeds 42 `
  --dataset_format mars `
  --out_root runs_task_level_bbh_tcs_useful_full `
  --resume_completed 1 `
  --resume_from_checkpoint 1
```

语义如下：

- 已完成目录被 `--resume_completed 1` 跳过；未完成目录从 checkpoint 继续。
- `training` 阶段从最近完成的 cursor 后继续；`epoch_evaluated` 与 `between_epochs` 会先完成相应 epoch 状态转换。
- 如果在并发 API batch 中途停止，正在飞行的小批调用可能重做；启用 `--candidate_reuse_recorded_rollouts 1` 时已记录 solver rollout 会被复用。
- checkpoint 会校验 reward、beam、candidate-eval、模型、数据路径等关键配置。不兼容时直接报错，**不会**在原目录静默从 step 0 重启。

若你要开新实验配置，应使用新的 `--out_root`；不要把不同配置继续写入旧 run 目录。

## 12. 常用命令

### 最小 smoke run

```powershell
python scripts/run_task_level_accuracy.py `
  --manifest configs/task_level_comparison.yaml `
  --tasks disambiguation_qa `
  --settings shared_baseline,shared_guarded_beam `
  --seeds 42 `
  --dataset_format mars `
  --out_root runs_task_level_smoke `
  --reward_mode coverage_useful_diversity `
  --optimizer_architecture teacher_critic_student `
  --optimizer_fallback_mode none `
  --epochs 1 `
  --train_size 20 `
  --val_size 20 `
  --test_size 40 `
  --update_every 10
```

### task-level 全量实验骨架

```powershell
python scripts/run_task_level_accuracy.py `
  --manifest configs/task_level_comparison.yaml `
  --benchmarks BBH,MMLU `
  --settings shared_baseline,shared_guarded_beam,bank_guarded_beam `
  --seeds 42 `
  --dataset_format mars `
  --out_root runs_task_level_accuracy `
  --reward_mode coverage_useful_diversity `
  --optimizer_architecture teacher_critic_student `
  --optimizer_fallback_mode none `
  --candidate_reuse_recorded_rollouts 1
```

Oracle Pareto 的预设实验 setting 为 `shared_oracle_pareto_tcs`，它固定使用 shared init、`coverage_useful_diversity`、`oracle_pareto`、`oracle_first`、TCS、`fixed_pool=50` 与 candidate eval batch 24。matched scalar 对照为 `shared_scalar_tcs_oracle_first`；两者除了 `candidate_selection_mode`（`scalar_reward` vs `oracle_pareto`）以外配置完全相同，且都使用 `oracle_first` validation：

```powershell
python scripts/run_task_level_accuracy.py `
  --manifest configs/task_level_comparison.yaml `
  --tasks disambiguation_qa `
  --settings shared_oracle_pareto_tcs `
  --seeds 42 `
  --dataset_format mars `
  --out_root runs_task_level_bbh_tcs_oracle_pareto
```

当前 `Config()` 的主要默认值为：`agents=5`、`epochs=2`、`train_size=200`、`val_size=100`、`test_size=200`、`update_every=10`、`beam_size=3`、`num_candidates_per_parent=2`、`candidate_eval_batch_size=20`、`candidate_eval_strategy=random`、`reward_mode=guarded_diversity`、`optimizer_architecture=teacher_critic_student`。

候选评估若追求更低方差，可显式改为：

```powershell
--candidate_eval_strategy fixed_pool `
--candidate_eval_pool_size 50 `
--candidate_eval_batch_size 20
```

### 汇总与外部比较

```powershell
python scripts/compute_experiment_metrics.py `
  --runs_root runs_task_level_accuracy `
  --out_csv runs_task_level_accuracy/experiment_metrics.csv `
  --out_md runs_task_level_accuracy/experiment_metrics.md

python scripts/analyze_student_failures.py `
  runs_task_level_accuracy/disambiguation_qa/shared_guarded_beam_seed42

python scripts/compare_external_accuracy.py `
  --mars_summary path/to/mars/summary.csv `
  --mad_results runs_task_level_accuracy/accuracy_results.jsonl `
  --out_csv comparison/mars_vs_mad_accuracy.csv `
  --out_md comparison/mars_vs_mad_accuracy.md
```

`compare_external_accuracy.py` 只读取用户提供的两个结果文件，不假设或读取 MARS 仓库目录结构。

## 13. 方法边界与报告注意事项

- 本项目优化 prompt，不训练模型权重；reward 不是 RL return。
- embedding diversity 是 reasoning-path diversity 的近似指标，不能单独证明真正的认知策略差异。
- candidate evaluation 有采样方差。重要结论应使用固定评估池或分层采样、多个 seed，并报告方差。
- `oracle_acc` 高而 `vote_acc` 低表示团队产生了正确少数路径，但当前聚合没有利用它；它不能直接证明最终团队更强。
- `weighted_vote` 不是学习得到的 verifier，`verifier_select` 当前仅回退到 majority。
- 成本统计仅用于报告，不参与任何训练限制或停止条件。
- 本仓库与 MARS 分开运行。manifest 覆盖不全时只能称为 subset comparison；同文件 train/val/test 时只能称为 paper-compatible setting，不能称为 strict split。
