# Method

本文档描述当前代码实现的方法。项目目标是构建一个多智能体 prompt 优化系统，使多个 LLM solver 在同一任务上形成可观察的推理路径差异，并用自动评估机制约束这种差异不以准确率、输出有效性或角色执行为代价。

## 1. 研究目标

多智能体推理常见做法是给多个 agent 设置不同角色，然后用投票或聚合得到最终答案。但如果多个 agent 的完整推理过程高度相似，那么表面上的“多角色”未必带来真实的信息增益。本项目关注的是：

- agent 之间的完整 reasoning trace 是否语义重复；
- 重复发生在哪些题、哪些 agent pair、哪些 prompt 行为上；
- 能否把这些重复案例自动转化为 prompt 优化信号；
- 新 prompt 是否真的改变了 solver 的解题路径；
- 这种改变是否仍然保持答案可靠性和输出格式有效性。

因此，系统把优化对象定义为“agent 的角色 prompt”，把观测对象定义为“完整 trace 与最终答案”，把主要多样性指标定义为“trace embedding overlap”。

## 2. 系统角色

系统包含三类模型角色。

### 2.1 Solver Agents

solver agents 是真正解题的模型实例。每个 agent 有自己的 active prompt。对于同一道题，系统并行调用所有 agent，要求输出：

```text
compact reasoning trace
FINAL_ANSWER: <answer>
```

MMLU 任务要求最终答案为 `A/B/C/D`；GSM8K 任务要求最终答案为数值或短答案。答案抽取逻辑位于 `multi_dataset_diverse_rl/utils.py`。

### 2.2 Prompt Optimizer

optimizer model 不直接解题，也不直接决定哪个 prompt 被采纳。它只根据系统整理出的窗口统计和行为案例，给指定 agent 的某个 parent prompt 生成候选 prompt。

optimizer 看到的信息包括：

- target agent id；
- parent prompt；
- target role preview；
- peer role previews；
- 窗口级 overlap / accuracy / invalid 统计；
- high-overlap cases；
- validity cases；
- random window cases；
- prompt 生成约束。

optimizer 不允许使用 gold answer、具体题目文本、选项文本、答案标签或样本 hash 作为 prompt 内容。

### 2.3 Evaluator

evaluator model 主要用于 local role execution 判断：候选 prompt 在小批量样本上运行后，evaluator 判断 target agent 的 trace 是否真的执行了候选 prompt 描述的角色程序。

这个判断只看单个 agent 的候选 prompt、role spec、trace 和 answer，不看 gold answer，也不比较其他 agent。这样可以避免 evaluator 把“答案正确”误当成“角色执行有效”。

系统还保留可选的 joint trace diversity evaluator，但默认 reward 不依赖它。默认多样性来自本地计算的 trace embedding。

## 3. 数据与任务抽象

每条数据被标准化为：

```python
{
    "question": "...",
    "answer": "..."
}
```

MMLU 推荐把题干和选项一起写进 `question`：

```text
Question: ...

Options:
A. ...
B. ...
C. ...
D. ...

Select the best option and output FINAL_ANSWER: <A/B/C/D>.
```

系统支持：

- `task_type=mmlu`
- `task_type=gsm8k`
- `task_type=auto`

`auto` 会根据题目和答案格式推断解析方式。

## 4. Agent 初始化

每个 agent 的状态由 `AgentState` 保存：

- `initial_prompt`
- `current_prompt`
- `prompt_beam`
- `history`
- `recent_homogeneity_flags`
- `accept_count`
- `reject_count`

初始化方式有两种：

### 4.1 Shared Init

所有 agent 从同一个 prompt 开始：

```text
You are a careful reasoning solver. Produce a compact, explicit reasoning trace, make your decision procedure visible, verify key logic, and give exactly one final answer.
```

这个设置用于检验系统是否能从同质起点中自动分化出不同角色。

### 4.2 Bank Init

每个 agent 从内置 prompt bank 中取一个不同初始角色。例如 MMLU 中包括：

- concept-first procedure
- contradiction-checking procedure
- boundary-and-scope procedure
- backward-validation procedure
- evidence-alignment procedure
- mechanism-first procedure

这个设置用于构造人工多角色 baseline。

## 5. Rollout

一次 rollout 处理一条题目。

输入：

```text
question
gold answer
active prompts for all agents
```

过程：

1. 所有 agent 并行调用 solver model。
2. 系统记录每个 agent 的 trace 和 extracted answer。
3. 对 extracted answers 做 majority vote。
4. 根据 task type 解析 gold answer。
5. 计算 individual correctness 和 vote correctness。
6. 检查每个 trace 是否有效。
7. 计算 trace embedding overlap 和 diversity。
8. 生成 high-overlap homogeneous cases 和 invalid validity cases。

输出被写入 `train_trace_history.jsonl` 或 prediction 文件中。

## 6. 输出有效性检查

系统使用 rule-based invalid checker，避免把无效输出当成有用多样性。一个 trace 会被判为 invalid，如果出现以下情况：

- trace 太短；
- 缺少 `FINAL_ANSWER:`;
- token 数太少；
- 无法抽取答案；
- bigram 重复比例过高。

这一步不依赖 LLM evaluator，保证基础格式约束稳定可复现。

重要设计：

> invalid trace 在 embedding overlap 计算中被视为与其他 trace 完全重叠。

也就是说，如果某个 agent 输出空泛、坏格式或重复文本，它不会获得 diversity bonus。这个设计阻止系统通过“破坏输出”来提高表面多样性。

## 7. Trace Embedding Diversity

默认多样性指标是完整 trace 的 embedding overlap。

步骤：

1. 对 trace 做空白归一化。
2. 如果 trace 很长，按 `trace_embedding_chunk_words` 分块，并用 `trace_embedding_chunk_overlap` 保留上下文重叠。
3. 使用 `sentence-transformers` 模型编码每个 chunk。
4. 对 chunk embedding 平均池化。
5. 对 pooled vector 归一化。
6. 计算 agent pair 之间的 cosine similarity。
7. 对 invalid pair 直接设 overlap 为 `1.0`。
8. 计算平均 overlap。
9. 得到 diversity：

```text
embedding_diversity = 1 - mean_embedding_overlap
```

默认 embedding 模型：

```text
BAAI/bge-small-en-v1.5
```

核心实现位于：

```text
TraceBeamSearchSystem.embedding_overlap_diagnostics
TraceBeamSearchSystem.compute_rollout_metrics
```

## 8. Homogeneous Cases

如果两个有效 trace 的 pair overlap 大于等于 `homogeneity_overlap_threshold`，系统认为它们构成一个 homogeneous case。

默认阈值：

```text
homogeneity_overlap_threshold = 0.55
```

每个 case 记录：

- `sample_hash`
- target / peer agent id
- pair overlap
- target trace preview
- peer trace preview
- target answer
- peer answer
- target prompt preview
- peer prompt preview
- team_correct

case 中只保留 trace preview，不把完整题目和完整答案交给 optimizer。这样做是为了让 optimizer 学习“行为模式”，而不是记忆具体样本。

## 9. Validity Cases

如果某个 agent 的 trace 被 rule invalid checker 判定无效，系统会构造 validity case。

validity case 包含：

- target agent id；
- trace preview；
- 是否存在可抽取答案；
- invalid reasons；
- target prompt preview。

当某个 agent 的 invalid rate 超过 `invalid_repair_rate_threshold` 时，系统会优先生成 validity-focused prompt candidates，先修复格式和有效性，再追求多样性。

## 10. 更新窗口

系统不是每个样本都更新 prompt，而是积累一个窗口。

窗口大小由：

```text
update_every
```

控制。默认值是 `10`。

窗口中保存：

- 最近样本的 traces；
- answers；
- prompts；
- rollout metrics；
- homogeneous cases；
- validity cases。

窗口满后，系统根据当前 reward mode 决定如何选择 agent。

### 10.1 默认模式：Overlap-Driven Update

当 `reward_mode=embedding_local_acc_invalid` 时，系统计算：

- per-agent overlap pressure；
- homogeneous case count；
- per-agent invalid rate；
- 最近窗口中每个 agent 的 homogeneity flags。

系统优先选择：

- invalid rate 超阈值的 agent；
- overlap pressure 最高的 agent；
- homogeneous case count 更多的 agent。

每个窗口通常更新 1 到 2 个 agent。

### 10.2 Accuracy-Only 模式

当 `reward_mode=accuracy_only` 时，系统不计算 embedding reward，也不追求 trace 多样性。它根据窗口中的 individual correctness 和 team correctness 选择错误较多的 agent，并让 optimizer 生成 accuracy repair prompt。

这个模式主要用于消融实验：对比“只优化准确率”和“同时约束 trace 多样性”的行为差异。

## 11. Case-Aware Candidate Generation

对每个待更新 agent，系统会把窗口诊断组织成 generation batches。

默认 diversity 模式包括：

```text
high_overlap_cases
mixed_window_cases
validity_focused_cases
```

含义：

- `high_overlap_cases`：重点处理 target agent 与 peer agent 高度相似的有效 trace pair。
- `mixed_window_cases`：加入随机窗口样本，降低只对最高 overlap case 过拟合的风险。
- `validity_focused_cases`：当输出无效或 fragile 时，优先修复格式和角色执行问题。

accuracy-only 模式包括：

```text
accuracy_error_cases
mixed_window_accuracy_cases
```

optimizer 对每个 beam parent 生成 `num_candidates_per_parent` 个候选 prompt。候选必须是可执行的角色程序，通常包含：

- role name；
- decision procedure；
- when to use；
- fallback strategy；
- anti-overlap rule 或 accuracy checks；
- validity checks；
- rationale；
- source batch type。

如果 optimizer 返回的候选不足，系统会补 fallback candidate，保证 beam search 可以继续。

## 12. Prompt Safety 与 Sanitize

候选 prompt 会经过 sanitize。

如果 prompt 中包含：

- `FINAL_ANSWER:` 模板；
- 明显复制的题目文本；
- 过多与当前问题重叠的长词；

系统会拒绝该候选或回退到 agent 初始 prompt。

这样可以防止 optimizer 把具体样本内容写进 prompt，造成数据泄漏或 prompt 过拟合。

## 13. Candidate Evaluation

候选 prompt 不会直接采纳。系统会在 `candidate_eval_batch_size` 个样本上评估它。

对某个 target agent 的候选 prompt，系统构造：

```text
eval_prompts = current peer prompts + candidate prompt for target agent
```

然后对每个 eval sample：

1. 所有 agent 重新作答。
2. 计算 team majority vote accuracy。
3. 计算完整 trace embedding diversity。
4. 检查 target agent trace 是否 invalid。
5. 计算 target agent 是否仍参与 high-overlap pair。
6. 统计解决了多少 baseline homogeneous cases。
7. 调用 evaluator 判断 target agent 是否真的执行了候选角色。

为了节省 API 成本，系统会缓存和复用已经记录过的 solver rollouts。相关指标包括：

- `solver_reuse_hits`
- `solver_reuse_misses`
- `solver_calls`
- `solver_reuse_hit_rate`

## 14. Reward

默认候选 prompt reward：

```text
reward =
  w_diversity      * embedding_diversity
+ w_local_validity * local_validity_mean
+ w_team_accuracy  * team_accuracy
+ w_invalid_score  * invalid_score
```

默认权重：

```text
w_diversity      = 0.5
w_local_validity = 0.2
w_team_accuracy  = 0.1
w_invalid_score  = 0.2
```

各项含义：

- `embedding_diversity`：候选 prompt 运行后团队 trace 的平均 embedding 多样性。
- `local_validity_mean`：target agent 是否实际执行候选 prompt 的角色程序。
- `team_accuracy`：小批量样本上 majority vote 是否正确。
- `invalid_score`：`1 - invalid_rate`。

accuracy-only 模式下：

```text
reward = team_accuracy
```

## 15. Evolutionary Beam Search

每个 agent 维护自己的 prompt beam。

一次更新：

1. 取当前 agent 的 beam。
2. 对每个 parent prompt 调用 optimizer 生成候选。
3. 把现有 beam prompt 也加入 candidate pool。
4. 并发评估所有候选。
5. 按 reward 降序排序。
6. 保留 top `beam_size`。
7. beam top-1 成为 agent 的 active prompt。
8. 记录 accepted / rejected、rank、reward 和指标。

beam item 结构：

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

这个设计让 prompt 搜索不是单步贪心替换，而是保留多个候选路径。

## 16. Beam Refresh

如果 `beam_refresh_each_epoch=1`，每个 epoch 结束时，系统会在新的 eval batch 上重新评估每个 agent 的 beam，并重新排序。

这样可以缓解某次小批量 candidate evaluation 的偶然性，也让旧候选有机会在新样本上重新竞争。

## 17. Validation 与 Early Stopping

每个 epoch 结束后，系统评估验证集。

默认 validation score：

```text
vote_acc + 0.2 * mean_embedding_diversity - 0.1 * mean_invalid_rate
```

如果 `reward_mode=accuracy_only`：

```text
validation score = vote_acc
```

当验证分数刷新，系统保存：

- `best_state.json`
- `best_prompts.json`

训练结束后，系统恢复 best prompts，再跑最终 test，写出：

- `selected_state.json`
- `test_final_predictions.jsonl`

因此最终测试不是简单使用最后一轮 prompt，而是使用验证集选择的 prompt。

## 18. 日志与可解释性

项目有意保留较多中间日志，方便分析 prompt 是如何改变的。

### 18.1 run_meta.json

记录：

- 模型配置；
- reward mode；
- embedding model；
- 初始 prompt；
- 完整 config。

### 18.2 history.json

epoch 级记录：

- train metrics；
- val metrics；
- optional test metrics；
- final selected epoch；
- early stopping 信息。

### 18.3 prompt_history.json

按 agent 记录：

- initial prompt；
- current prompt；
- prompt hash；
- prompt beam；
- update / refresh event。

### 18.4 update_logs.jsonl

每个候选 prompt 一行，包含：

- reward；
- embedding diversity；
- mean embedding overlap；
- local validity；
- team accuracy；
- invalid rate；
- homogeneous case count；
- resolved case count；
- new homogeneous case count；
- 是否进入 beam；
- rank；
- generation batch type；
- prompt preview。

### 18.5 trace history

`train_trace_history.jsonl` 和 `test_trace_history.jsonl` 保留每道题的 agent trace、answer、invalid 信息和 case 信息，用于离线复盘。

## 19. 与普通 Self-Refine / Prompt Search 的区别

本项目不是让一个模型根据分数反复改写一个 prompt。它有几个关键区别：

1. 优化对象是多 agent 团队中的单个 agent prompt。
2. 优化信号来自 agent 之间的 trace 关系，而不是单个答案分数。
3. optimizer 只提出候选，采纳由独立 evaluation 决定。
4. 多样性是完整 trace 的语义多样性，不是 prompt 文本差异。
5. 输出有效性和角色执行被显式纳入 reward。
6. 无效输出不会获得 diversity 奖励。
7. 使用 beam 保留多个 prompt 演化分支。

## 20. 当前实现边界

当前主实现聚焦 `trace_embedding` diversity。

保留的 taxonomy 相关脚本和 `taxonomies/mmlu_reasoning_family_taxonomy.json` 可用于离线分析策略族、证明实验或扩展 reward，但默认主训练循环不依赖 taxonomy reward。

当前系统还没有解决所有问题：

- optimizer 仍可能生成泛化不足或过长的 prompt；
- embedding overlap 只能近似衡量 trace 语义相似；
- local role execution 依赖 evaluator model，可能有判断噪声；
- 小批量 candidate evaluation 会有采样方差；
- 多 agent 并发调用成本较高；
- 对不同任务，合适的 prompt bank 和 reward 权重可能需要重新调参。

## 21. 推荐实验设计

建议按以下顺序使用：

1. 跑 `shared_baseline`：确认同 prompt 多 agent 的自然同质化程度。
2. 跑 `bank_baseline`：确认人工多角色 prompt 能否提高 trace diversity。
3. 跑 `shared + evolutionary_beam`：观察系统能否从同质起点自动分化。
4. 跑 `accuracy_only` 消融：确认只优化准确率时是否会忽略 trace 差异。
5. 调整 reward weights：检查 diversity、accuracy、invalid rate 的 trade-off。
6. 分析 `update_logs.jsonl`：查看被采纳 prompt 是否真的降低 overlap。
7. 抽查 `train_trace_history.jsonl` 和 `test_final_predictions.jsonl`：人工确认 trace 是否发生可解释变化。

## 22. 算法概览

伪代码：

```text
initialize agents with shared prompt or prompt bank
initialize each agent beam with its initial prompt

for epoch in epochs:
    shuffle train data

    for each training example:
        run all agents on the question
        extract answers and compute majority vote
        check invalid traces
        compute trace embedding overlap
        build homogeneous and validity cases
        append rollout to update window

        if step % update_every == 0 and window is ready:
            diagnose the window
            select 1-2 agents to update

            for each selected agent:
                build case generation batches
                for each parent prompt in the agent beam:
                    ask optimizer for candidate prompts
                add existing beam prompts to candidate pool
                evaluate candidates on a small batch
                score candidates with reward
                keep top beam_size prompts
                set top-1 as active prompt

            clear update window

    optionally refresh all beams
    evaluate validation set
    save best prompts by validation score
    early stop if no improvement

restore best prompts
evaluate final test set
write states, histories, predictions, and logs
```

## 23. 代码定位

主要实现：

```text
multi_dataset_diverse_rl/system.py
```

关键函数：

- `solve_once`：单个 agent 解题。
- `compute_rollout_metrics`：答案、invalid、embedding diversity 等指标。
- `embedding_overlap_diagnostics`：trace embedding overlap 诊断。
- `_build_homogeneous_cases`：构造高重叠 case。
- `_build_validity_cases`：构造无效输出 case。
- `_window_overlap_diagnosis`：窗口级同质化诊断。
- `_build_case_generation_batches`：把诊断转为 optimizer 输入批次。
- `propose_candidates`：调用 optimizer 生成候选 prompt。
- `evaluate_candidate_prompt`：小批量评估候选。
- `_candidate_reward`：计算 reward。
- `update_prompt_with_beam`：evolutionary beam 更新。
- `refresh_all_prompt_beams`：epoch 末 beam 重新评估。
- `evaluate_dataset`：val/test 评估与 prediction 文件写出。

入口：

```text
multi_dataset_diverse_rl/cli.py
```

配置：

```text
multi_dataset_diverse_rl/config.py
```
