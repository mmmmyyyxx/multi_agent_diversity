# 多智能体推理策略多样化 Prompt RL

这个项目实现了一个基于 textual gradient 与轻量 bandit/RL 更新的多智能体 Prompt 训练框架。它的目标不是直接最大化准确率，而是让一组同构 LLM agents 在推理策略族上形成更分散、更互补的角色。

当前版本使用 LLM judge 给每条推理轨迹判定 reasoning family，并基于 family 分布计算 reward。此前的 skeleton / 骨架相似度链路已移除，不再参与 reward、更新选择或日志记录；完整推理文本分开保存在 `train_trace_history.jsonl` 与 `test_trace_history.jsonl`，用于人工分析。

## 目录结构

```text
multi_dataset_diverse_rl/
  config.py      # 命令行参数与默认配置
  utils.py       # 数据读取、答案解析、family 指标等工具函数
  policy.py      # BanditPolicy 与 AgentState
  system.py      # 训练主逻辑、诊断、rewriter、日志与评估
  cli.py         # python -m multi_dataset_diverse_rl.cli 入口
scripts/
  prepare_mmlu_jsonl.py       # MMLU 数据转换
  run_ablation_matrix.py      # 消融实验批量运行
  run_testonly_baselines.py   # 仅测试不训练的 shared/bank baseline 运行与汇总
  analyze_ablation.py         # 汇总消融结果
  plot_training_dynamics.py   # 绘制训练动态
  plot_ablation_results.py    # 绘制消融对比图
  plot_ablation_with_baselines.py  # A/B/C/D + test-only baseline 统一对比图
run.ps1 / run.sh              # 训练快捷脚本
baseline_run.ps1 / .sh        # baseline 评估快捷脚本
method.md                     # 方法定义与公式说明
```

## 环境准备

安装依赖：

```bash
pip install -r requirements.txt
```

设置 OpenAI API Key：

```powershell
$env:OPENAI_API_KEY = "your_key"
```

如果使用兼容 OpenAI API 的代理服务，可额外设置：

```powershell
$env:OPENAI_BASE_URL = "your_base_url"
```

## 数据格式

训练集和测试集均为 JSONL，每行至少包含：

- `question`
- `answer`

示例：

```json
{"question": "What is 2+2?", "answer": "4"}
```

任务类型由 `--task_type` 控制：

- `auto`：根据答案和题面自动判断。
- `gsm8k`：数值答案任务。
- `mmlu`：A/B/C/D 选择题任务。

MMLU 数据转换示例：

```bash
python scripts/prepare_mmlu_jsonl.py --dataset_name cais/mmlu --dataset_config all --train_split dev --test_split test --out_train mmlu_train.jsonl --out_test mmlu_test.jsonl
```

## 快速运行

PowerShell 快捷训练：

```powershell
.\run.ps1 auto train.jsonl test.jsonl
```

等价核心命令：

```powershell
python -m multi_dataset_diverse_rl.cli `
  --task_type auto `
  --train_path train.jsonl `
  --test_path test.jsonl `
  --agents 4 `
  --init_mode shared `
  --epochs 2 `
  --update_every 5 `
  --candidate_eval_batch_size 3 `
  --train_size 200 `
  --test_size 100
```

Baseline 只评估不训练：

```powershell
.\baseline_run.ps1 shared auto test.jsonl
```

或直接运行：

```powershell
python -m multi_dataset_diverse_rl.cli `
  --task_type auto `
  --baseline_only 1 `
  --init_mode shared `
  --test_path test.jsonl `
  --test_size 100
```

## 关键参数

**模型参数**

- `--model`：solver agents 使用的模型，默认 `gpt-4o-mini`。
- `--critic_model`：group diagnosis / family judge 使用的模型。
- `--rewriter_model`：prompt rewriter 使用的模型。
- `--max_tokens`、`--critic_max_tokens`、`--rewriter_max_tokens`：各阶段最大输出长度。
- `--temperature`、`--critic_temperature`、`--rewriter_temperature`：各阶段采样温度。

**动态分类参数**

- `--family_expansion_model`：审核模型，用于判断新 family 是否应被接纳，默认 `deepseek-v4-pro`。
- `--family_expansion_enabled`：是否允许扩充 family 集合（1/0）。关闭时会强制映射回已有 family。
- `--family_taxonomy_path`：动态 taxonomy JSON 路径，新 family 被接纳后会写入该文件。
- `--use_dual_family_labels`：是否启用主策略 + 子策略判别，默认 `1`。设为 `0` 时退回旧的单策略映射。
- `--primary_family_weight`：主策略在单条轨迹策略分布中的权重，默认 `0.7`。
- `--secondary_family_weight`：子策略在单条轨迹策略分布中的权重，默认 `0.3`。
- `--same_major_family_weight`：两个策略属于同一主类但具体方法不同时的相似度，默认 `0.5`。
- `--macro_diversity_weight`：团队多样性中跨主类分化的权重，默认 `0.5`，其余权重给主类内子方法分化。

当前内置基础 reasoning family 按 `分类.md` 组织为 7 个主类和 36 个具体方法。系统不再使用 `other` 分类；空输出、异常输出或未知标签会被映射到最接近的有效策略。旧标签如 `contradiction_proof`、`elimination_comparison`、`backward_verification`、`invariant_symmetry` 会自动映射到新的规范标签。

**训练参数**

- `--agents`：agent 数量。
- `--init_mode shared|bank`：共享初始 prompt 或从内置 prompt bank 初始化。
- `--epochs`：训练轮数。
- `--train_size` / `--test_size`：读取的数据量上限。
- `--update_every`：每多少个 step 尝试一次 prompt 更新。
- `--candidate_eval_batch_size`：候选 prompt 小批评估样本数。

**Reward 参数**

- `--lambda_diversity`：鼓励 family 多样性和个体 family 新颖性。
- `--lambda_homogeneity`：惩罚同 family 重叠。
- `--lambda_invalid_trace`：惩罚空输出、过短输出、格式缺失、重复严重等无效轨迹。

当前 reward 只使用 family-level 信号与基础无效输出防护，不使用 skeleton/文本相似度。

## 训练流程

每个训练样本大致执行以下步骤：

1. 所有 agents 用各自当前 prompt 解题，输出完整 trace 与 `FINAL_ANSWER:`。
2. 系统调用 LLM judge 批量查看多条完整 trace，默认输出 `primary_family`、`secondary_family` 和 `reasoning_summary`。`reasoning_summary` 用一到两句话概括推理思路，后续 group critic、rewriter 和新标签审核都使用该摘要。若出现新标签则由审核模型决定接纳或映射，并持久化 taxonomy。
3. 根据主/子策略加权分布计算 team diversity、family homogeneity 与每个 agent 的 reward。
4. 将当前样本加入窗口；当 `step % update_every == 0` 且窗口填满时尝试更新。
5. Group critic 基于窗口级摘要生成 `group_summary` 与 `target_role_hints`。
6. Rewriter 为被选中的 agent 生成候选 prompt。
7. Bandit 在 `keep_current + candidates` 中采样动作。
8. 系统用小批样本比较当前 prompt 与候选 prompt 的 mean reward，决定 `keep`、`accept` 或 `reject`。
9. 写入 step 日志、update 日志、trace 日志和 prompt 历史。

## 输出文件

默认输出目录为 `runs_tg_rl/`，也可通过 `--out_dir` 指定。

- `run_meta.json`：配置、初始化方式、初始 prompt hash、模型参数。
- `history.json`：每个 epoch 的 train/test 汇总指标。
- `prompt_history.json`：每个 agent 的 prompt 初始化、更新、采纳、拒绝和 sanitize 事件。
- `update_logs.jsonl`：prompt 更新尝试的压缩日志，包括诊断摘要、候选 hash、动作、决策、错误信息。
- `train_step_logs.jsonl`：每个训练 step 的 family 指标、reward 摘要和 update 摘要。
- `train_trace_history.jsonl`：训练阶段完整推理轨迹。
- `test_trace_history.jsonl`：测试阶段完整推理轨迹。
- `reasoning_summary_history.jsonl`：轻量回溯索引，按样本保存每个 agent 的 `primary_family`、`secondary_family`、`family_resolution`、`secondary_family_resolution`、family 分布、`reasoning_summary`、`trace_hash` 和题目短摘；需要查看完整原文时可用 `trace_hash` 回到 trace history 中定位。
- `family_taxonomy.json`：动态 family taxonomy（默认写入，路径可由 `--family_taxonomy_path` 指定）。
- `test_epoch*_predictions.jsonl`：每个测试样本的答案、投票结果和 family 指标。
- `last_state.json` / `best_state.json`：agent 状态、bandit 参数和 prompt 历史快照。

不会再生成 `skeleton_history.jsonl`。

## 指标说明

- `mean_family_diversity`：平均 family 分布多样性；双策略模式下同时考虑跨主类分化和主类内子方法分化。
- `mean_family_homogeneity_rate`：平均 family 同质率；双策略模式下是 pairwise 加权策略重叠度。
- `vote_acc`：多数投票答案准确率，仅用于评估，不直接作为 diversity reward。
- `mean_invalid_trace_penalty`：训练 step 日志中的无效轨迹惩罚均值。
- `mean_llm_direct_diversity_score`：LLM 直接给出的组级多样性评分（仅记录，不参与 reward）。

日志会记录 `primary_family_labels`、`secondary_family_labels`、`reasoning_summaries`、`agent_family_distributions`、`primary_family_counts`、`weighted_family_distribution`、`major_family_distribution`、`team_major_family_diversity` 和 `team_intra_family_diversity`。

## 消融实验

运行：

```powershell
python scripts/run_ablation_matrix.py --workspace . --out_root runs_abcd --task_type auto --train_path mmlu_train.jsonl --test_path mmlu_test.jsonl --epochs 2 --agents 5
```

当前脚本默认启用两组设置：

- `A_shared_no_div`：shared 初始化，`lambda_diversity/lambda_homogeneity/lambda_invalid_trace` 全为 0。
- `B_shared_div`：shared 初始化，开启 family diversity reward。

`scripts/run_ablation_matrix.py` 中保留了 `bank` 初始化设置的注释行，如需 A/B/C/D 四组完整矩阵，可取消 `C_bank_no_div` 与 `D_bank_div` 两行注释。

运行后会生成：

- `runs_abcd/abcd_runs.jsonl`
- `runs_abcd/abcd_runs.csv`
- `runs_abcd/ablation_summary.csv`
- `runs_abcd/ablation_summary.md`

### 新增：测试集直评 baseline（不训练）

在已有 A/B/C/D 结果基础上，增加两组 baseline：

- `E_shared_testonly`：`init_mode=shared`，`baseline_only=1`
- `F_bank_testonly`：`init_mode=bank`，`baseline_only=1`

运行：

```powershell
python scripts/run_testonly_baselines.py --workspace . --out_root runs_abcd --task_type auto --test_path test.jsonl --test_size 100 --agents 5
```

该脚本会在 `runs_abcd/` 下生成：

- `baseline_runs.jsonl` / `baseline_runs.csv`
- `abcd_plus_baselines.jsonl` / `abcd_plus_baselines.csv`
- `ablation_summary_with_baselines.csv` / `ablation_summary_with_baselines.md`
- `ablation_with_baselines_diversity_panel.png`
- `ablation_with_baselines_homogeneity_panel.png`
- `ablation_with_baselines_behavior_panel.png`
- `ablation_with_baselines_trace_summary_panel.png`：在同一张图中并排比较完整 trace 与 `reasoning_summary` 的 cosine diversity/similarity。

## 结果分析与绘图

汇总已有 runs：

```powershell
python scripts/analyze_ablation.py --runs_root runs_abcd --out_csv runs_abcd/ablation_summary.csv --out_md runs_abcd/ablation_summary.md
```

绘制训练动态：

```powershell
python scripts/plot_training_dynamics.py --base_dir runs_abcd --out_dir runs_abcd/figures
```

绘制消融结果：

```powershell
python scripts/plot_ablation_results.py --csv runs_abcd/ablation_summary.csv --out_dir runs_abcd/figures
```

## 可修改超参数及物理意义

**模型与生成**

- `--model`：solver agent 的模型，决定解题轨迹本身的能力和风格。
- `--critic_model`：family judge、group diagnosis 使用的模型，决定策略分类和诊断质量。
- `--rewriter_model`：prompt 改写模型，决定候选 prompt 的质量。
- `--family_expansion_model`：新 family 审核模型，只在 judge 提出未见标签时调用。
- `--max_tokens` / `--critic_max_tokens` / `--rewriter_max_tokens`：分别限制 solver、critic、rewriter 输出长度。
- `--temperature` / `--critic_temperature` / `--rewriter_temperature`：分别控制 solver、critic、rewriter 采样随机性。

**分类与多样性统计**

- `--use_dual_family_labels`：是否使用主策略 + 子策略。`1` 表示每条轨迹保留 top-2 策略；`0` 表示退回单策略。
- `--primary_family_weight`：主策略权重，默认 `0.7`，越大越强调轨迹的 dominant strategy。
- `--secondary_family_weight`：子策略权重，默认 `0.3`，越大越强调混合推理轨迹中的辅助策略。
- `--same_major_family_weight`：同主类不同子方法的相似度，默认 `0.5`。越大，系统越认为“同一大类内部的方法仍然相似”。
- `--macro_diversity_weight`：跨主类多样性在 `D_family` 中的权重，默认 `0.5`。越大越鼓励 agents 分散到不同主类；越小越鼓励同一主类内部的细粒度分化。
- `--family_expansion_enabled`：是否允许训练过程中接纳新 family。关闭后未知标签会映射到已有策略。
- `--family_taxonomy_path`：动态 family taxonomy 保存路径。

动态扩展审核只查看触发新标签的单个 agent 完整 trace，不使用其他 agents 或组级分布；审核上下文会以 `family_resolution.review_context="single_agent_trace"`、`trace_hash`、`trace_length` 等字段写入 `reasoning_summary_history.jsonl`，便于回溯。

**Reward**

- `--lambda_diversity`：多样性奖励强度。越大，越鼓励团队覆盖不同策略。
- `--lambda_homogeneity`：同质性惩罚强度。越大，越惩罚策略重叠。
- `--lambda_invalid_trace`：无效轨迹惩罚强度。越大，越压制空输出、过短输出、格式缺失或重复严重的轨迹。

**训练与更新**

- `--agents`：参与协作的 agent 数量。
- `--init_mode`：初始 prompt 方式，`shared` 表示共享初始 prompt，`bank` 表示从内置 prompt bank 初始化。
- `--epochs`：训练轮数。
- `--train_size` / `--test_size`：训练集/测试集读取上限。
- `--update_every`：每多少个训练 step 尝试一次 prompt 更新；同时决定 homogeneity window 的实际长度。
- `--candidate_eval_batch_size`：评估候选 prompt 时使用的小批样本数。
- `--bandit_lr`：bandit 更新步长。越大，采纳/拒绝反馈对动作偏好的影响越快。
- `--baseline_momentum`：bandit baseline 的动量。越大，历史 reward 影响越长。
- `--seed`：随机种子。

**稳定性与重试**

- `--max_retries`：普通 LLM 调用最大重试次数。
- `--retry_sleep`：重试基础等待时间。
- `--transient_retry_forever`：是否对临时 API 错误持续重试。
- `--max_transient_retries`：临时错误最大重试次数，`0` 表示不设上限。
- `--max_retry_backoff`：重试等待时间上限。
- `--llm_call_logging`：是否在控制台打印每次 LLM 调用的 start/ok/retry/failed 日志，默认开启，便于定位卡住阶段。
- `--llm_call_timeout`：单次 LLM 请求超时时间（秒），默认 `120`；超时后会打印 retry 日志并按重试策略继续。

## 注意事项

- 运行前必须设置 `OPENAI_API_KEY`。
- `update_every` 会同步决定 homogeneity window 的实际长度。
- `train_trace_history.jsonl` / `test_trace_history.jsonl` 会保存完整模型输出，文件可能较大，也可能包含题面相关内容。
- `reasoning_summary_history.jsonl` 不保存完整 trace，只保存策略摘要和 hash，适合快速回溯推理路径分类；原文仍以 trace history 为准。
- Reward 不直接使用准确率；准确率主要用于观察多样性训练是否破坏任务表现。
- Skeleton 相关代码和日志已移除，如果旧实验目录中还存在旧 `skeleton_history.jsonl`，那是历史产物，不代表当前实现。

方法公式与抽象流程见 [method.md](method.md)。
