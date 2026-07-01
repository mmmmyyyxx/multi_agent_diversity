# 多智能体 Trace 多样性 Prompt 优化框架

这个项目是一个用于研究“多智能体推理是否真的形成不同解题路径”的实验框架。它让多个 solver agent 同时回答同一道题，记录每个 agent 的完整推理 trace 和最终答案，然后用 trace embedding 衡量不同 agent 之间的语义重叠。系统会把高重叠、无效输出、错误答案等行为证据反馈给一个 optimizer model，让它为指定 agent 生成新的角色 prompt。新的 prompt 不是立即采纳，而是进入 evolutionary beam search，由小批量评估决定是否保留。

换句话说，本项目研究的不是简单地“让 prompt 写得更花”，而是让一个多智能体团队在保持答案可靠性的同时，产生可观察、可评估、真正不同的推理过程。

## 项目在做什么

核心问题：

- 多个 LLM agent 使用同一个或相似 prompt 时，是否会给出高度相似的推理 trace？
- 能不能自动发现这种同质化，并把具体案例反馈给 prompt optimizer？
- 新 prompt 能不能让某个 agent 采用不同的解题程序，同时不牺牲多数投票准确率？
- 多样性指标如何避免被无效输出、格式错误、空泛 trace 或重复文本“刷分”？

系统把一次样本处理成以下对象：

1. 同一道题由多个 agent 并行作答。
2. 每个 agent 输出一个 compact reasoning trace，并以 `FINAL_ANSWER:` 给出最终答案。
3. 系统抽取答案，做 majority vote，并与 gold answer 比较。
4. 系统检查 trace 是否有效。
5. 对有效 trace 计算 sentence-transformer embedding，并计算 agent pair 的 cosine overlap。
6. 对高 overlap 的有效 trace pair 生成 homogeneous cases。
7. 对无效 trace 生成 validity cases。
8. 当窗口累计到 `update_every` 个样本后，系统选择最需要更新的 agent。
9. optimizer 根据 parent prompt、窗口统计和案例证据生成候选 prompt。
10. 候选 prompt 在小批量样本上评估，通过 reward 排序进入 beam。
11. 每个 agent 的 beam top-1 成为下一轮 active prompt。

默认 reward 是：

```text
reward =
  reward_weight_diversity      * embedding_diversity
+ reward_weight_local_validity * local_validity_mean
+ reward_weight_team_accuracy  * team_accuracy
+ reward_weight_invalid_score  * invalid_score
```

默认权重为 `0.5 / 0.2 / 0.1 / 0.2`。其中 `embedding_diversity = 1 - mean_pairwise_embedding_overlap`，`invalid_score = 1 - invalid_rate`。

## 当前仓库内容

仓库已清理掉数据集和历史实验输出。当前保留的是代码、脚本、taxonomy、prompt 模板和方法文档。

```text
multi_dataset_diverse_rl/
  cli.py       # 主运行入口，负责加载数据、训练、验证、测试和早停
  config.py    # 所有命令行参数和默认配置
  policy.py    # AgentState 与每个 agent 的 prompt beam 状态
  system.py    # 多智能体 rollout、trace 评估、optimizer 调用、beam search 主逻辑
  utils.py     # 数据读取、答案抽取、策略族工具、指标工具

scripts/
  prepare_mmlu_data.py          # 从 HuggingFace MMLU 准备项目 JSONL 数据
  sample_jsonl_splits.py        # 从已有数据生成小规模、去重、均衡 split
  run_experiments.py            # 批量运行 baseline / beam 设置
  compute_experiment_metrics.py # 汇总 run 输出为 CSV / Markdown
  plot_*.py                     # 绘制训练和对比图
  prove_* / analyze_*           # 策略树、多样性证明和离线分析相关脚本

taxonomies/
  mmlu_reasoning_family_taxonomy.json

README.md
method.md
requirements.txt
```

兼容入口：

```text
multi_dataset_diverse_prompt_rl.py
```

它只是转发到 `python -m multi_dataset_diverse_rl.cli`。

## 支持的数据格式

主程序读取 JSONL，每行至少需要能抽取出 `question` 和 `answer`。常见字段名已经在 `utils.py` 中兼容：

- 问题字段：`question`、`input`、`query`、`problem`
- 答案字段：`answer`、`output`、`target`、`label`、`response`

MMLU 示例：

```json
{"question": "Question: ...\n\nOptions:\nA. ...\nB. ...\nC. ...\nD. ...\n\nSelect the best option and output FINAL_ANSWER: <A/B/C/D>.", "answer": "B", "subject": "abstract_algebra"}
```

GSM8K 示例：

```json
{"question": "Natalia sold clips to 48 friends...", "answer": "#### 72"}
```

`--task_type auto` 会根据题目和答案格式推断任务类型；也可以显式指定 `--task_type mmlu` 或 `--task_type gsm8k`。

## 安装

建议使用虚拟环境：

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

项目使用 OpenAI-compatible Chat Completions 接口：

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://your-compatible-endpoint/v1"
```

Windows PowerShell：

```powershell
$env:OPENAI_API_KEY="..."
$env:OPENAI_BASE_URL="https://your-compatible-endpoint/v1"
```

如果 solver 和 evaluator 使用不同 endpoint，可以通过参数指定环境变量名：

```bash
--solver_api_key_env SOLVER_API_KEY \
--solver_base_url_env SOLVER_BASE_URL \
--evaluator_api_key_env EVALUATOR_API_KEY \
--evaluator_base_url_env EVALUATOR_BASE_URL
```

## 准备数据

仓库当前不包含数据集。可以用脚本从 HuggingFace MMLU 转成项目 JSONL：

```bash
python scripts/prepare_mmlu_data.py \
  --dataset_name cais/mmlu \
  --dataset_config all \
  --train_split validation \
  --val_split dev \
  --test_split test \
  --out_train mmlu_train.jsonl \
  --out_val mmlu_val.jsonl \
  --out_test mmlu_test.jsonl \
  --train_limit 200 \
  --val_limit 150 \
  --test_limit 200 \
  --balanced 1
```

也可以从已有 train/val/test 源文件重新采样：

```bash
python scripts/sample_jsonl_splits.py \
  --train_source full_train.jsonl \
  --val_source full_val.jsonl \
  --test_source full_test.jsonl \
  --out_train mmlu_train.jsonl \
  --out_val mmlu_val.jsonl \
  --out_test mmlu_test.jsonl \
  --train_size 200 \
  --val_size 150 \
  --test_size 200
```

## 最小运行

单次训练运行：

```bash
python -m multi_dataset_diverse_rl.cli \
  --task_type mmlu \
  --train_path mmlu_train.jsonl \
  --val_path mmlu_val.jsonl \
  --test_path mmlu_test.jsonl \
  --out_dir runs_trace_beam/shared_beam_seed42 \
  --agents 5 \
  --init_mode shared \
  --epochs 3 \
  --update_every 10 \
  --candidate_eval_batch_size 10 \
  --beam_size 3 \
  --num_candidates_per_parent 2 \
  --reward_mode embedding_local_acc_invalid \
  --embedding_model BAAI/bge-small-en-v1.5
```

只跑 baseline，不更新 prompt：

```bash
python -m multi_dataset_diverse_rl.cli \
  --task_type mmlu \
  --test_path mmlu_test.jsonl \
  --out_dir runs_trace_beam/shared_baseline_seed42 \
  --agents 5 \
  --init_mode shared \
  --baseline_only 1
```

`init_mode` 有两种：

- `shared`：所有 agent 从同一个 shared prompt 开始，适合观察自动分化能力。
- `bank`：agent 从内置 prompt bank 中取不同初始角色，适合作为人工多角色 baseline。

## 批量实验

`scripts/run_experiments.py` 会调用主 CLI 并把不同设置写入同一个输出根目录。当前脚本默认启用 baseline 设置；可以按需要编辑 `SETTINGS`。

```bash
python scripts/run_experiments.py \
  --out_root runs_trace_beam \
  --task_type mmlu \
  --train_path mmlu_train.jsonl \
  --val_path mmlu_val.jsonl \
  --test_path mmlu_test.jsonl \
  --agent_model deepseek-chat \
  --optimizer_model deepseek-v4-flash \
  --evaluator_model deepseek-v4-flash \
  --agents 5 \
  --epochs 3 \
  --seeds 42
```

## 主要参数

模型：

- `--agent_model`：solver agent 使用的模型。
- `--optimizer_model`：生成候选 prompt 的模型。
- `--evaluator_model`：判断 local role execution 和可选 joint diversity 的模型。

数据：

- `--train_path`、`--val_path`、`--test_path`
- `--train_size`、`--val_size`、`--test_size`
- `--val_split_ratio`：没有 `val_path` 时，从 train 中切出验证集。

多智能体：

- `--agents`：agent 数量。
- `--init_mode shared|bank`
- `--shared_prompt`：shared 初始化使用的基础 prompt。

搜索与更新：

- `--search_mode evolutionary_beam`
- `--beam_size`：每个 agent 保留的 prompt 数。
- `--num_candidates_per_parent`：每个 beam parent 生成几个候选。
- `--update_every`：每多少个训练样本触发一次更新窗口。
- `--candidate_eval_batch_size`：候选 prompt 小批量评估样本数。
- `--beam_refresh_each_epoch`：每个 epoch 结束后是否重新评估 beam。

多样性与有效性：

- `--reward_mode embedding_local_acc_invalid`：默认模式，用 trace embedding 多样性、local validity、team accuracy 和 invalid score 共同打分。
- `--reward_mode accuracy_only`：消融模式，只按团队准确率更新。
- `--homogeneity_overlap_threshold`：高重叠 case 阈值。
- `--embedding_model`：trace embedding 模型。
- `--trace_embedding_chunk_words`、`--trace_embedding_chunk_overlap`：长 trace 分块参数。

并发与重试：

- `--eval_solver_call_concurrency`
- `--candidate_eval_concurrency`
- `--train_rollout_concurrency`
- `--llm_call_timeout`
- `--transient_retry_forever`
- `--max_retry_backoff`

## 输出文件

每个 run 的 `out_dir` 会包含：

```text
run_meta.json                 # 配置、模型、初始 prompt
history.json                  # epoch 级训练、验证、测试指标
prompt_history.json           # 每个 agent 的 prompt beam 和更新事件
update_logs.jsonl             # 每个候选 prompt 的 reward、rank、是否进入 beam
train_step_logs.jsonl         # 训练 step 级指标
train_trace_history.jsonl     # 训练题目的 agent trace、answer、case
test_trace_history.jsonl      # val/test trace 记录
val_epochN_predictions.jsonl  # 每轮验证集预测
test_final_predictions.jsonl  # 最终测试集预测
last_state.json               # 最后状态
best_state.json               # 验证集选择的最好状态
selected_state.json           # 最终恢复 best prompt 后的测试状态
best_prompts.json             # early stopping 选中的 prompt
```

关键指标：

- `vote_acc`：多 agent majority vote 准确率。
- `mean_embedding_diversity`：平均 trace embedding 多样性。
- `mean_embedding_overlap`：平均 trace embedding 重叠。
- `mean_invalid_rate`：无效 trace 比例。
- `reward`：候选 prompt 综合得分。
- `local_validity_mean`：候选 prompt 是否被 solver 实际执行。
- `homogeneous_case_count`：目标 agent 仍参与多少高重叠 case。
- `resolved_case_count`：候选 prompt 解决了多少原高重叠 case。
- `new_homogeneous_case_count`：候选 prompt 引入了多少新高重叠 case。

## 分析实验结果

汇总 run：

```bash
python scripts/compute_experiment_metrics.py \
  --runs_root runs_trace_beam \
  --out_csv runs_trace_beam/experiment_metrics.csv \
  --out_md runs_trace_beam/experiment_metrics.md
```

绘图脚本可以按需要使用：

```bash
python scripts/plot_current_mmlu_visuals.py --root runs_trace_beam --out_dir runs_trace_beam/figures
python scripts/plot_experiment_results.py --csv runs_trace_beam/experiment_metrics.csv --out_dir runs_trace_beam/figures
```

不同 plot 脚本对输入文件有不同假设，建议先看脚本头部的 argparse。

## 方法设计要点

本项目最重要的设计不是“让 prompt 越不同越好”，而是把多样性放进一个受约束的优化目标：

- trace 必须有效，必须能抽取最终答案。
- agent 必须实际执行候选 prompt 描述的角色程序。
- 多样性来自完整 trace embedding，而不是 prompt 文本差异。
- 无效 trace 在 overlap 中按完全重叠处理，防止通过坏输出获得虚假多样性。
- optimizer 只能提出候选，不直接决定采纳。
- beam search 通过小批量评估保留更可靠的候选。
- early stopping 用验证集选择最终 prompt，而不是默认使用最后一轮。

更完整的方法说明见 [method.md](method.md)。

## 当前限制

- 项目依赖 LLM API，实验成本与并发设置强相关。
- trace embedding 默认使用 `BAAI/bge-small-en-v1.5`，首次运行可能需要下载模型。
- 当前核心 diversity metric 是 trace embedding；taxonomy 相关脚本保留用于离线证明和扩展分析。
- optimizer 生成 prompt 时仍可能产生过泛、过长或不可执行的角色描述，因此系统加入了 prompt sanitize、local validity 和 invalid check。
- `scripts/` 中包含一些针对历史证明实验的分析脚本，运行前应确认输入目录是否存在。

## 推荐工作流

1. 准备小规模、去重、均衡的 JSONL 数据。
2. 先跑 `shared` 和 `bank` baseline，确认模型和答案抽取正常。
3. 跑 `shared` beam，观察是否能从同 prompt 自动分化。
4. 汇总 `history.json`、`update_logs.jsonl` 和 prediction 文件。
5. 对比 accuracy、embedding diversity、invalid rate 和 prompt history。
6. 如果 invalid rate 升高，调高 invalid 权重或降低 optimizer 温度。
7. 如果 diversity 没有变化，检查 homogeneous cases 是否被触发、candidate_eval_batch 是否过小、embedding 模型是否正常加载。
