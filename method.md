# Method

本文档描述当前仓库实现的方法。核心目标是让多个 LLM solver agent 在同一任务上形成可观察、可评估的推理路径差异，同时用准确率、输出有效性和平票诊断约束这种差异。

需要特别明确：本项目方法是 **case-aware evolutionary prompt search**，不是严格 reinforcement learning。系统没有训练模型参数，也没有策略梯度；它通过 rollout 诊断、案例驱动 prompt generation、candidate evaluation 和 per-agent beam search 来演化 prompt。

## 1. 问题定义

多 agent 推理常见做法是给多个 agent 不同角色，再用 majority vote 得到最终答案。但如果 agent 的完整 reasoning trace 高度相似，那么表面上的多角色未必带来真实互补性。

本项目关注：

- 同一题中，不同 agent 的 trace 是否语义重叠。
- 哪些题、哪些 agent pair、哪些 prompt 行为导致重叠。
- 能否把高重叠、无效输出、错误答案等案例转成 prompt 优化信号。
- 新 prompt 是否真正改变 target agent 的解题程序。
- 多样性提升是否会损害 majority vote accuracy。

## 2. 系统组件

系统包含三类模型角色：

- Solver agents：真正解题，输出 compact reasoning trace 和 `FINAL_ANSWER: <answer>`。
- Prompt optimizer：根据窗口统计和案例证据提出候选 prompt。
- Evaluator：判断 target agent 是否执行了候选 prompt 描述的角色程序。

Optimizer 不直接决定采纳。候选 prompt 必须经过 evaluation 和 beam ranking。

## 3. TaskSpec 任务抽象

任务逻辑集中在 `multi_dataset_diverse_rl/tasks.py`：

```python
@dataclass
class TaskSpec:
    name: str
    parse_gold: Callable[[str, Optional[str]], str]
    extract_pred: Callable[[Optional[str], Optional[str]], str]
    match_answer: Callable[[str, str], bool]
    format_question: Optional[Callable[[dict], str]] = None
```

当前支持：

- `mmlu`
- `gsm8k`
- `bbh`
- `auto`

`utils.py` 中的旧接口仍然保留，例如 `parse_gold`、`extract_pred_answer_by_task`、`infer_task_type`，但内部委托给 `TaskSpec`。`system.py` 中 gold parsing、prediction extraction 和 answer comparison 都通过 `self.task_spec` 完成。

## 4. MMLU / GSM8K / BBH

MMLU：

- gold 归一化为 `A/B/C/D`。
- prediction 支持 `FINAL_ANSWER: A`、`Answer: (A)`、`option A` 等。
- matching 使用归一化后的 exact match。

GSM8K：

- gold 兼容 `#### 72`。
- prediction 优先读 `FINAL_ANSWER:`、`Answer:`、`The answer is` 后的数字。
- 没有显式标记时才 fallback 到最后一个数字。
- 数字会去逗号，`3.0` 归一化为 `3`。

BBH：

- 独立 parser，不 fallback 到 GSM8K。
- prediction 优先读最后一行 `FINAL_ANSWER: <content>`，其次 `Answer: <content>`，否则取最后一个非空行。
- 不盲目取最后一个数字，避免污染 boolean、date、logical deduction 等 BBH 任务。
- yes/no/true/false 归一化为可比较别名。
- `(A)`、`A.`、`option A` 归一化为 `a`。
- 支持 gold aliases list，例如 `["yes", "true"]`。

`auto` 会识别 MMLU/GSM8K；无法确定时按 BBH 风格处理，这是为了避免未知任务被 GSM8K 数字 parser 误解析。

## 5. 数据格式

CLI 的 `build_dataset(raw_records, dataset_format)` 支持两类格式。

`legacy`：

- question: `question`, `input`, `query`, `problem`
- answer: `answer`, `output`, `target`, `label`, `response`

`mars`：

- question: `question`, `input`, `query`, `problem`, `prompt`
- answer: `answer`, `target`, `gold`, `gold_answer`, `label`, `output`
- task/subtask: `task`, `task_name`, `category`, `subject`, `bbh_task`

如果无法抽取 question 或 answer，系统会抛出带 record index 的错误，便于定位坏数据。

## 6. Rollout

一次 rollout 输入：

```text
question
gold answer
active prompts for all agents
```

流程：

1. 对所有 agent 并行调用 solver。
2. 用 `TaskSpec.extract_pred` 抽取每个 agent 的答案。
3. 用 majority vote 聚合团队答案。
4. 用 `TaskSpec.parse_gold` 解析 gold。
5. 用 `TaskSpec.match_answer` 计算 individual correctness 和 vote correctness。
6. 检查 trace 是否 invalid。
7. 计算 trace embedding overlap 和 diversity。
8. 构造 homogeneous cases 和 validity cases。

## 7. Majority Vote Diagnostics

旧 majority vote 在平票时返回最早出现的答案，会偏向低编号 agent。现在系统使用 `majority_vote_with_diagnostics`：

```json
{
  "vote_answer": "A",
  "vote_tie": true,
  "tie_candidates": ["A", "B"],
  "vote_counts": {"A": 1, "B": 1},
  "tie_break_method": "random"
}
```

tie-break 策略：

- `first`：保留旧行为。
- `random`：默认，用 `cfg.seed + question_hash` 做 deterministic random。
- `abstain`：平票返回空答案。

这些字段会进入 rollout metrics、prediction 日志、`train_step_logs.jsonl` 和最终汇总。`compute_experiment_metrics.py` 会输出 `vote_tie_rate`。

## 8. Trace 有效性

系统用 rule-based invalid checker 防止无效输出获得 diversity bonus。常见 invalid 原因：

- trace 太短。
- 缺少 `FINAL_ANSWER:`。
- token 数过少。
- 无法抽取最终答案。
- bigram 重复比例过高。

关键约束：

```text
invalid trace 在 embedding overlap 计算中按完全重叠处理
```

也就是说，坏格式、空泛、重复输出不会带来虚假的 diversity 提升。

## 9. Trace Embedding Diversity

默认 diversity metric 是完整 trace 的 embedding overlap：

1. 归一化空白。
2. 长 trace 按 `trace_embedding_chunk_words` 分块。
3. 使用 `sentence-transformers` embedding model 编码。
4. 对 chunk embedding 平均池化。
5. 计算 agent pair 的 cosine similarity。
6. 得到 `mean_embedding_overlap`。
7. 计算：

```text
embedding_diversity = 1 - mean_embedding_overlap
```

默认 embedding model：

```text
BAAI/bge-small-en-v1.5
```

## 10. Homogeneous Cases

如果两个有效 trace 的 pair overlap 高于 `homogeneity_overlap_threshold`，系统构造 homogeneous case。

case 包含：

- sample hash
- target / peer agent id
- pair overlap
- trace preview
- answer
- prompt preview
- team correctness

case 中只保留 preview，不把完整题目、完整答案或 gold 交给 optimizer，降低数据泄漏和 prompt 过拟合风险。

## 11. Validity Cases

如果某个 agent 输出 invalid trace，系统构造 validity case：

- target agent id
- trace preview
- 是否存在可抽取答案
- invalid reasons
- prompt preview

当某个 agent invalid rate 超过 `invalid_repair_rate_threshold`，系统会优先生成 validity-focused candidates。

## 12. 更新窗口

系统不是每个样本都更新 prompt，而是用 `update_every` 积累窗口。

窗口保存：

- traces
- extracted answers
- prompts
- rollout metrics
- homogeneous cases
- validity cases

窗口满后，系统选择需要更新的 agent：

- `embedding_local_acc_invalid` 和 `guarded_diversity`：主要看 overlap pressure、homogeneous cases、invalid rate。
- `accuracy_only`：主要看 individual/team error。

## 13. Candidate Generation

对每个待更新 agent，系统把窗口诊断组织成 generation batches：

- `high_overlap_cases`
- `mixed_window_cases`
- `validity_focused_cases`
- `accuracy_error_cases`
- `mixed_window_accuracy_cases`

Optimizer 为每个 beam parent 生成 `num_candidates_per_parent` 个候选 prompt。候选 prompt 应该是可执行角色程序，通常包含 role name、decision procedure、fallback strategy、accuracy checks、validity checks 和 anti-overlap rule。

候选 prompt 会经过 sanitize。如果包含 `FINAL_ANSWER:` 模板、明显复制题目、或过多复用样本词汇，会被拒绝或回退。

## 14. Candidate Evaluation

候选 prompt 不直接采纳。对 target agent，系统构造：

```text
candidate_prompts = current peer prompts + candidate prompt replacing target agent
```

在 `guarded_diversity` 下，还会在同一 eval batch 上构造 baseline：

```text
baseline_prompts = current active prompts, target agent not replaced
```

每个 eval sample 计算：

- team accuracy
- embedding diversity
- invalid rate
- mean embedding overlap
- target local validity
- solver reuse metrics

Candidate eval 支持：

- `random`
- `fixed_pool`
- `stratified`

训练开始时会构建 `candidate_eval_pool`：优先从 validation data 采样，否则从 train data 采样。`fixed_pool` 在固定 seed 下可复现。`candidate_eval_repeats > 1` 时，CLI 合并多个 batch，最终 reward 和 metrics 取平均。

## 15. Reward Modes

### 15.1 guarded_diversity

默认 reward mode。它以 baseline-relative 的方式约束 candidate：

```text
acc_delta = candidate_team_acc - baseline_team_acc
div_delta = candidate_embedding_diversity - baseline_embedding_diversity
invalid_delta = candidate_invalid_rate - baseline_invalid_rate
```

如果准确率下降超过 `accuracy_guard_epsilon`：

```text
reward = -1.0 + acc_delta - reward_weight_invalid_delta * max(0, invalid_delta)
```

否则：

```text
reward =
    candidate_team_acc
  + reward_weight_div_delta * div_delta
  + reward_weight_local_validity * local_validity
  - reward_weight_invalid_delta * max(0, invalid_delta)
```

这让多样性只在准确率 guard 通过时获得奖励。

### 15.2 embedding_local_acc_invalid

旧 reward mode，保留用于兼容和历史实验：

```text
reward =
  reward_weight_diversity      * embedding_diversity
+ reward_weight_local_validity * local_validity
+ reward_weight_team_accuracy  * team_accuracy
+ reward_weight_invalid_score  * invalid_score
```

### 15.3 accuracy_only

消融模式：

```text
reward = team_accuracy
```

该模式不加载 embedding model，也不追求 trace diversity。

## 16. Evolutionary Beam Search

每个 agent 维护自己的 prompt beam：

```json
{
  "id": "...",
  "prompt": "...",
  "score": 0.0,
  "metrics": {},
  "parent_id": "...",
  "generation": 1
}
```

一次更新：

1. 取 target agent 当前 beam。
2. 对每个 parent prompt 生成候选。
3. 把现有 beam prompt 也加入候选池。
4. 并发评估候选。
5. 按 reward 降序排序。
6. 保留 top `beam_size`。
7. beam top-1 成为 active prompt。
8. 写入 update logs 和 prompt history。

这保留了 TraceBeamSearchSystem / evolutionary beam search 主流程，只增强任务解析、投票、reward 和评估稳定性。

## 17. Validation 和 Early Stopping

每个 epoch 后评估 validation set。

`accuracy_only` 使用：

```text
validation score = vote_acc
```

其他模式使用：

```text
validation score = vote_acc + 0.2 * mean_embedding_diversity - 0.1 * mean_invalid_rate
```

刷新 best score 时保存：

- `best_state.json`
- `best_prompts.json`

训练结束后恢复 best prompts，再跑 final test。

## 18. 实验协议

`scripts/run_experiments.py` 默认运行四个 setting：

- `shared_baseline`
- `bank_baseline`
- `shared_guarded_beam`
- `bank_guarded_beam`

推荐命令：

```bash
python scripts/run_experiments.py --datasets mmlu,bbh --seeds 42,43,44
```

目录结构：

```text
runs_trace_beam/{dataset}/{setting}_seed{seed}
```

每个 setting 使用自己的 reward mode；`--force_reward_mode` 可覆盖全部 setting。

## 19. 汇总指标

`scripts/compute_experiment_metrics.py` 递归读取多 dataset 子目录，并按 `dataset/setting` 计算 mean/std。

核心输出：

- `latest_test_vote_acc`
- `latest_test_embedding_diversity`
- `latest_test_invalid_rate`
- `vote_tie_rate`
- `solver_calls`
- `solver_reuse_hit_rate`

如果提供 `--mars_result_path`，会输出：

- `vs_mars_delta_acc`
- `vs_mars_delta_diversity`

## 20. 日志

`update_logs.jsonl` 记录每个 candidate prompt：

- reward
- embedding diversity
- local validity
- team accuracy
- invalid rate
- baseline/candidate guarded metrics
- accuracy guard 是否通过
- eval strategy、pool size、batch size、repeats
- solver reuse metrics
- beam rank / accepted

`train_step_logs.jsonl` 记录每个训练 step：

- vote correctness
- vote answer
- vote tie diagnostics
- embedding diversity
- invalid rate
- update summary

Prediction 文件记录：

- vote answer
- gold
- vote correctness
- vote tie diagnostics
- each agent trace / answer / invalid status

## 21. 实现边界

- 系统优化的是 prompt，不训练模型参数。
- Reward 是候选排序信号，不是严格 RL return。
- Trace embedding diversity 是语义相似度近似，不等同于人工策略标签。
- Candidate evaluation 仍有采样方差，需要多 seed 和固定/分层评估池。
- BBH parser 只做规范化和 exact/numeric/alias matching，不做模糊语义匹配，避免虚高 accuracy。

## 22. 代码定位

主要实现：

```text
multi_dataset_diverse_rl/system.py
```

关键函数：

- `solve_once`
- `compute_rollout_metrics`
- `embedding_overlap_diagnostics`
- `_build_homogeneous_cases`
- `_build_validity_cases`
- `_window_overlap_diagnosis`
- `_build_case_generation_batches`
- `propose_candidates`
- `evaluate_candidate_prompt`
- `_candidate_reward_guarded`
- `update_prompt_with_beam`
- `refresh_all_prompt_beams`
- `evaluate_dataset`

任务解析：

```text
multi_dataset_diverse_rl/tasks.py
```

入口和数据格式：

```text
multi_dataset_diverse_rl/cli.py
```

配置：

```text
multi_dataset_diverse_rl/config.py
```
