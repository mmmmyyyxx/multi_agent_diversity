# 方法

本项目实现了面向多智能体推理的**案例感知进化式提示搜索**。它并非严格的强化学习：不更新模型权重，不执行策略梯度步骤，奖励仅用于在每个智能体的束搜索内对候选提示进行排序。

一句话目标：

```text
利用提示进化诱导智能体形成互补的推理路径；
在训练期间奖励能够填补团队盲点的有用多样性；
默认使用多数投票，在推理/评估时可选地使用多样性感知的加权投票。
```

## 1. 当前实现摘要

活跃的训练循环是 `TraceBeamSearchSystem`，采用进化束搜索：

1. 多个求解智能体使用各自的当前提示回答同一问题。
2. 系统记录答案、轨迹、有效性、投票结果、预言机覆盖率和轨迹多样性。
3. 每 `update_every` 步，一次以奖励为导向的窗口诊断会选择一到两个智能体进行提示更新。
4. 候选提示由配置的提示进化架构生成。
5. 候选者在相同的候选评估批次上，与当前的活跃团队进行对比评估。
6. 候选者按奖励排序，每个被更新的智能体保留一个大小为 `beam_size` 的提示束。
7. 验证挑选出最佳提示；最终测试使用恢复的最佳提示。

## 2. 模型角色与命名

CLI 保留了历史字段名 `agent_model`、`optimizer_model` 和 `evaluator_model`，但它们目前的角色是：

- `agent_model`：求解器执行轨迹模型。它为每个智能体回答训练/验证/测试问题。
- `optimizer_model`：提示进化生成模型。它用于一次性优化器以及教师-评论家-学生生成端的调用：教师、教师重写、学生、学生 JSON 重试和学生 JSON 修复。
- `evaluator_model`：评估者/审计者模型。它用于 TCS 评论家以及可选的联合轨迹多样性评估器。
- `embedding_model`：本地句子转换器，用于轨迹嵌入多样性诊断。

因此，`optimizer_model` 不再意味着“直接编写所有提示的单一整体优化器”。在默认架构中，它是用于结构化教师-评论家-学生提示进化管线的生成模型。

`run_meta.json` 将这些映射记录在 `model_role_map` 中。

## 3. 任务与数据集支持

任务解析集中在 `multi_dataset_diverse_rl/tasks.py` 中，通过 `TaskSpec` 实现：

```python
@dataclass
class TaskSpec:
    name: str
    parse_gold: Callable[[Any, Optional[str]], str]
    extract_pred: Callable[[Optional[str], Optional[str]], str]
    match_answer: Callable[[str, str], bool]
    format_question: Optional[Callable[[dict], str]] = None
```

支持的任务类型：

- `mmlu`
- `gsm8k`
- `bbh`
- `auto`

当提供 `--answer_format` 时，将使用 `multi_dataset_diverse_rl/answer_formats.py`。支持的格式：

- `option_letter`
- `boolean`
- `yes_no`
- `valid_invalid`
- `numeric`
- `free_text`

数据集格式：

- `legacy`：问题键为 `question/input/query/problem`；答案键为 `answer/output/target/label/response`。
- `mars`：问题键为 `question/input/query/problem/prompt`；答案键为 `answer/target/gold/gold_answer/label/output`；元数据键为 `task/task_name/category/subject/bbh_task`。

任务级比较使用本仓库自己的清单文件 `configs/task_level_comparison.yaml`。它不依赖于本地的 MARS 仓库。

## 4. 执行轨迹与聚合

对于每个样本，所有活跃提示回答同一问题：

```text
question + active_prompt_i -> trace_i + answer_i
```

执行轨迹日志包括：

- 每个智能体的答案及正确性
- 投票答案和投票正确性
- 预言机正确性，即至少有一个智能体正确
- 无效轨迹标志
- 轨迹嵌入多样性和重叠度
- 投票平局诊断
- 可选的加权投票诊断

求解器轨迹应包含：

```text
FINAL_ANSWER: <answer>
```

### 多数投票

默认聚合方式是多数投票。平局被显式记录：

```json
{
  "vote_tie": true,
  "tie_candidates": ["A", "B"],
  "vote_counts": {"A": 1, "B": 1},
  "tie_break_method": "random"
}
```

平局打破选项：

- `first`：旧版的首个答案行为。
- `random`：使用 `seed + question_hash` 的确定性随机；这是默认设置。
- `abstain`：平局时返回空投票答案。

### 加权投票

`--aggregation_mode weighted_vote` 是可选的。它利用有效性和轨迹独立性来减少冗余答案的影响：

```text
weight_i = reliability_i * validity_i * independence_i
independence_i = min(max(1 - per_agent_overlap_i, 0), 0.5)
score(answer) = sum(weight_i for agents predicting answer)
```

可靠性目前是均匀的；这是一种轻量级的聚合规则，而非学习得到的验证器。

## 5. 轨迹有效性与多样性

无效检查器基于规则。当轨迹过短、缺少 `FINAL_ANSWER:`、词元过少、无法提取答案或出现高度重复时，它会将轨迹标记为无效。

轨迹多样性根据完整的推理轨迹计算，而非提示文本：

```text
embedding_diversity = 1 - mean_embedding_overlap
```

对于有用多样性，新颖性仅在目标轨迹有效且正确时才被计入。

## 6. 提示进化架构

默认架构是：

```bash
--optimizer_architecture teacher_critic_student
```

旧版的一次性优化器仍然可用：

```bash
--optimizer_architecture one_shot
```

### 教师-评论家-学生

TCS 是当前结构化的提示进化路径：

1. **教师**使用抽象的窗口诊断信息创建苏格拉底式指导性问题。它不直接编写候选提示。
2. **评论家**在学生看到之前审计教师的问题。它拒绝泛泛的、无根据的、泄露信息的、硬编码的、仅表面多样性或损害准确性的问题。
3. 如果被拒绝，**教师重写**修订问题，评论家再次审计。
4. **学生**根据批准的教师问题生成候选提示。
5. 学生候选仍然经过现有的模式检查、冗余检查、候选评估、奖励排序和束选择。

TCS 不使用预定义的任务特定角色。投票失败不是默认的教师目标；提示质量、任务对齐、目标智能体准确性和有用多样性才是目标。

### 学生 JSON 稳定性

期望学生的输出是严格的 JSON。默认模式为紧凑模式：

```bash
--student_candidate_schema_mode compact
```

如果学生输出为空或格式错误：

```text
初始学生调用
-> 使用更严格的仅 JSON 指令重试学生，最多 student_json_max_retries 次
-> 如果非空的格式错误 JSON 仍然失败，则调用 JSON 修复
-> 仅当恢复出有效的 JSON 对象时才继续
```

当前默认设置：

```text
student_json_retry_on_parse_fail = True
student_json_max_retries = 5
student_json_repair_enabled = True
student_json_repair_max_tokens = 1200
student_json_repair_temperature = 0.0
student_candidate_prompt_max_chars = 900
student_candidate_max_chars_per_field = 320
```

JSON 修复并非模板回退。它仅修复已生成的学生内容的语法，不会凭空创造提示想法。

重要的诊断信息：

- `student_raw_response_empty`
- `student_json_parse_failed`
- `student_json_retry_attempted`
- `student_json_retry_succeeded`
- `student_json_repair_attempted`
- `student_json_repair_succeeded`
- `student_failure_stage`
- `student_candidate_count_raw`
- `student_candidate_count_final`

最终的学生失败应由 `student_candidate_count_final == 0` 来判断，而不仅仅是某个中间的部分失败阶段。

## 7. 候选评估

对于目标智能体，候选评估比较：

```text
baseline_prompts  = 当前活跃提示
candidate_prompts = 将目标智能体替换为候选提示后的当前活跃提示
```

这使得候选评估是相对于基线的。

每个样本的候选评估日志包括：

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

定义：

```text
target_useful_diversity =
    target_trace_novelty
  * target_agent_correct
  * target_valid

rescue =
    基线投票错误
    且目标智能体正确
```

在当前奖励中，`rescue` 和 `rescue_rate` 仅用于诊断。

候选评估策略：

- `random`
- `fixed_pool`
- `stratified`

为获得更低的方差，推荐：

```bash
--candidate_eval_strategy fixed_pool
--candidate_eval_pool_size 100
--candidate_eval_batch_size 20
```

## 8. 奖励模式

代码默认设置是：

```bash
--reward_mode guarded_diversity
```

对于当前的 TCS 有用多样性实验，显式运行：

```bash
--reward_mode coverage_useful_diversity
```

可用模式：

- `guarded_diversity`：目标智能体准确率约束加上轨迹多样性增量和无效率惩罚。
- `coverage_useful_diversity`：目标智能体准确率加上预言机覆盖率增益和有用多样性，并带有无效约束。
- `accuracy_only`：消融模式，按更新后目标智能体自身的准确率对候选进行排序，同时仍记录团队投票准确率。

### 带约束的多样性

`guarded_diversity` 使用目标智能体准确率作为约束和主要准确率信号：

```text
acc_delta     = candidate_target_acc - baseline_target_acc
vote_delta    = candidate_team_acc - baseline_team_acc
div_delta     = candidate_embedding_diversity - baseline_embedding_diversity
invalid_delta = candidate_invalid_rate - baseline_invalid_rate

if candidate_target_acc < baseline_target_acc - effective_accuracy_guard_epsilon:
    reward = -1.0 + acc_delta - weight_invalid_delta * max(0, invalid_delta)
else:
    reward =
        weight_target_accuracy * candidate_target_acc
      + weight_div_delta       * div_delta
      - weight_invalid_delta   * max(0, invalid_delta)
```

团队投票准确率被记录用于诊断，但不是奖励基础。

### 覆盖率有用多样性

`coverage_useful_diversity` 是测试有用多样性的推荐模式：

```text
coverage_delta =
    candidate_oracle_acc - baseline_oracle_acc

invalid_guard_passed =
    candidate_invalid_rate <= baseline_invalid_rate + invalid_guard_epsilon

if not invalid_guard_passed:
    reward = -1.0
else:
    reward =
        effective_weight_target_accuracy * candidate_target_accuracy
      + effective_weight_coverage        * coverage_delta
      + effective_weight_useful_diversity * useful_diversity
```

活跃组件：

- `candidate_target_accuracy`
- `coverage_delta`
- `useful_diversity`
- `invalid_guard`

以下指标被记录但不直接优化：

- `candidate_team_accuracy`
- `vote_delta`
- `rescue_rate`
- `rescue_useful_diversity`

旧的 `coverage_rescue_diversity` 模式已被移除。

## 9. 阶段自适应奖励调度

默认调度是：

```bash
--reward_schedule_mode phase_adaptive
```

早期更新使用更强的多样性/有用多样性权重，特别是当智能体从共享提示开始时。随着提示的分化和已接受更新的积累，调度将权重移回目标智能体准确率，并使用更严格的准确率约束。

记录的有效字段包括：

- `effective_weight_target_accuracy`
- `effective_weight_div_delta`
- `effective_weight_coverage`
- `effective_weight_useful_diversity`
- `effective_accuracy_guard_epsilon`
- `reward_phase_progress`
- `reward_diversity_need`
- `reward_unique_prompt_ratio`

## 10. 束搜索

每个智能体拥有自己的提示束。在更新时：

1. 现有的束提示充当父级。
2. 每个父级向优化器架构请求新的候选。
3. 现有的束提示也保留在候选池中。
4. 所有池中的项目都经过候选评估。
5. 按奖励排名前 `beam_size` 的项目成为新的束。
6. 束中的第一名成为活跃提示。

这意味着一个新的候选可能被生成，但如果现有的束提示在评估批次上得分更高，它就不会成为活跃提示。

`best_prompts.json` 存储由验证选出的提示，并为最终测试恢复。

## 11. 回退与空操作进化

默认情况下模板回退是禁用的：

```bash
--optimizer_fallback_mode none
```

如果启用，模板回退是一种工程安全机制，应单独报告。清晰的方法声明应使用 `none`。

空操作进化提前停止是启用的：

```bash
--no_effective_evolution_stop_enabled 1
```

如果重复的更新尝试产生的优化器候选太少，且没有活跃提示更改，进化将停止，最终评估仍然运行。

## 12. 验证与最终指标

对于 `coverage_useful_diversity`，验证分数是：

```text
0.4 * vote_acc
+ 0.3 * oracle_acc
+ 0.2 * mean_useful_diversity
- 0.2 * mean_invalid_rate
```

最终数据集指标包括：

- `vote_acc`
- `majority_vote_acc`
- `weighted_vote_acc`
- `mean_individual_acc`
- `best_individual_acc`
- `oracle_acc`
- `aggregation_gap`
- `rescue_available_rate`
- `correct_disagreement_rate`
- `mean_useful_diversity`
- `vote_tie_rate`
- `mean_invalid_rate`

解读：

- `oracle_acc`：是否有某个智能体给出了正确答案。
- `vote_acc`：聚合是否使用了正确答案。
- `aggregation_gap`：还有多少正确的少数信息未被使用。
- `mean_useful_diversity`：多样性是否来自有效/正确的轨迹。

## 13. 任务级准确率导出

任务级运行导出标准化的行以便后续比较：

```bash
python scripts/run_task_level_accuracy.py \
  --manifest configs/task_level_comparison.yaml \
  --benchmarks BBH,MMLU \
  --settings shared_baseline,shared_guarded_beam,bank_guarded_beam \
  --seeds 42 \
  --dataset_format mars \
  --out_root runs_task_level_accuracy
```

对于 TCS 有用多样性运行，添加：

```bash
--reward_mode coverage_useful_diversity
--optimizer_architecture teacher_critic_student
```

输出：

```text
runs_task_level_accuracy/accuracy_results.jsonl
```

行包括任务元数据、设置、种子、准确率指标、预言机/挽救诊断以及 LLM 调用/词元统计。

与 MARS 的比较应在外部进行。MAD 不读取 MARS 仓库。按 `task_id` 单独连接，并至少报告：

- MARS 准确率
- MAD 投票准确率
- MAD 平均智能体准确率
- MAD 最佳智能体准确率

## 14. 成本统计

每次运行写入：

```text
llm_calls.jsonl
cost_summary.json
```

成本字段：

- `solver_calls`
- `optimizer_calls`
- `evaluator_calls`
- `total_llm_calls`
- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- `estimated_cost`
- `latency_seconds`

这些字段仅用于报告。它们不约束、停止、排序或规范化训练。

## 15. 关键代码位置

核心文件：

```text
multi_dataset_diverse_rl/config.py
multi_dataset_diverse_rl/cli.py
multi_dataset_diverse_rl/system.py
multi_dataset_diverse_rl/tasks.py
multi_dataset_diverse_rl/answer_formats.py
scripts/run_task_level_accuracy.py
scripts/analyze_student_failures.py
scripts/compute_experiment_metrics.py
scripts/compare_external_accuracy.py
```

`TraceBeamSearchSystem` 中的重要方法：

- `compute_rollout_metrics`
- `_window_update_diagnosis`
- `select_reward_agents_for_update`
- `propose_candidates_teacher_critic_student`
- `generate_student_candidates`
- `retry_student_candidates_json_only`
- `repair_student_json_response`
- `_candidate_reward_guarded`
- `_candidate_reward_coverage_useful_diversity`
- `evaluate_candidate_prompt`
- `update_prompt_with_beam`
- `_summarize_rollout_rows`
- `evaluate_dataset`

## 16. 方法边界

- 系统优化的是提示，而非模型权重。
- 奖励是一个候选排序信号，而非强化学习中的回报。
- 轨迹嵌入多样性是对推理路径多样性的近似。
- 候选评估存在采样方差；为得出可靠结论，应使用固定池、分层抽样和多个种子。
- `weighted_vote` 是一种轻量级的聚合规则，而非学习得到的验证器。
- 任务级比较的完整性取决于 `configs/task_level_comparison.yaml`；不完整的清单覆盖意味着只能进行子集比较。
## Checkpoint Resume

Each non-baseline training run writes `training_checkpoint.json` in its run directory. `--resume_from_checkpoint 1` restores agent prompts, prompt beams, accepted/rejected counts, history, prompt history, cost summary, best-validation state, and the current epoch cursor.

Checkpoint stages:

- `training`: resume from the saved epoch cursor and continue the remaining training examples.
- `epoch_evaluated`: validation for the epoch has been computed; resume by finalizing best-prompt and early-stop bookkeeping.
- `between_epochs`: the epoch is fully committed; resume from the next epoch or final test.

For task-level experiments, use both:

```bash
--resume_completed 1
--resume_from_checkpoint 1
```

`resume_completed` skips complete run directories and rebuilds summary files. `resume_from_checkpoint` continues incomplete run directories. The resume granularity is batch/epoch-level; an in-flight API batch can still be repeated after an interruption, and recorded solver rollouts are reused when enabled.

If an existing checkpoint was written with incompatible resume-critical settings, resume fails fast and prints the mismatched fields. It does not silently restart from step 0 in the same run directory.
