# 多智能体推理策略多样化 Prompt RL

这个项目实现了一个基于 textual gradient 与轻量 bandit/RL 更新的多智能体 Prompt 训练框架。目标不是直接最大化准确率，而是让一组同构 LLM agents 在推理策略族上形成更分散、更互补的角色。

当前版本使用 LLM judge 为每条推理轨迹判定 reasoning family，并基于 family 分布计算 reward。完整推理文本保存在 `train_trace_history.jsonl` 与 `test_trace_history.jsonl`，策略摘要保存在 `reasoning_summary_history.jsonl`，用于回溯和分析。

## 目录结构

```text
multi_dataset_diverse_rl/
  config.py      # 命令行参数与默认配置
  utils.py       # 数据读取、答案解析、family 指标等工具函数
  policy.py      # BanditPolicy 与 AgentState
  system.py      # 训练主逻辑、诊断、rewriter、日志与评估
  cli.py         # python -m multi_dataset_diverse_rl.cli 入口
scripts/
  prepare_mmlu_data.py          # MMLU 数据转换
  run_experiments.py            # 顺序运行四个实验设置
  analyze_experiments.py        # 读取已有 run，统一计算指标并画图
  compute_experiment_metrics.py # 汇总实验指标
  plot_experiment_results.py    # 四个实验设置统一结果图
  plot_experiment_dynamics.py   # 训练和测试动态曲线
  plot_training_comparison.py   # 训练组对比图
run.ps1 / run.sh                # 训练快捷脚本
baseline_run.ps1 / .sh          # 只测试不训练的快捷脚本
method.md                       # 方法定义与公式说明
```

## 环境准备

安装依赖：

```bash
pip install -r requirements.txt
```

设置 API：

```powershell
$env:OPENAI_API_KEY = "your_key"
$env:OPENAI_BASE_URL = "your_base_url"  # 可选，兼容 OpenAI API 的中转服务
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
python scripts/prepare_mmlu_data.py --dataset_name cais/mmlu --dataset_config all --train_split dev --test_split test --out_train mmlu_train.jsonl --out_test mmlu_test.jsonl
```

## 快速运行

核心训练命令：

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

只测试不训练：

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
- `--critic_model`：family judge、group diagnosis 使用的模型。
- `--rewriter_model`：prompt rewriter 使用的模型。
- `--family_expansion_model`：新 family 审核模型，默认 `deepseek-v4-pro`。
- `--max_tokens` / `--critic_max_tokens` / `--rewriter_max_tokens`：各阶段最大输出长度。

**分类与摘要参数**

- `--family_taxonomy_path`：动态 family taxonomy 文件路径。
- `--family_expansion_enabled`：是否允许训练中接纳新 family。
- `--use_dual_family_labels`：是否启用主策略 + 子策略判别，默认 `1`。
- `--primary_family_weight`：主策略权重，默认 `0.7`。
- `--secondary_family_weight`：子策略权重，默认 `0.3`。
- `--same_major_family_weight`：同一主类不同子策略的相似度，默认 `0.5`。
- `--macro_diversity_weight`：跨主类多样性在 `D_family` 中的权重，默认 `0.5`。
- `--min_summary_words`：reasoning summary 的最小词数要求。
- `--max_summary_tokens`：reasoning summary 的最大 token 数。项目优先使用 `tiktoken` 计数和截断，若不可用则退回单词数近似。

**训练与 reward 参数**

- `--agents`：agent 数量。
- `--init_mode shared|bank`：共享初始 prompt 或从内置 prompt bank 初始化。
- `--epochs`：训练轮数。
- `--train_size` / `--test_size`：读取的数据量上限。
- `--update_every`：每多少个 step 尝试一次 prompt 更新。
- `--candidate_eval_batch_size`：候选 prompt 小批评估样本数。
- `--lambda_diversity`：鼓励 family 多样性和个体策略新颖性。
- `--lambda_homogeneity`：惩罚策略同质化。
- `--lambda_invalid_trace`：惩罚空输出、过短输出、格式缺失、重复严重等无效轨迹。

## 训练流程

1. 所有 agents 用各自当前 prompt 解题，输出完整 trace 与 `FINAL_ANSWER:`。
2. 系统对每个 agent 的完整 trace 并发调用 single-trace LLM judge，输出 `primary_family`、`secondary_family`、`reasoning_summary`、`strategy_steps`、`distinctive_features`、`evidence_spans` 和 `confidence`。
3. `reasoning_summary` 是详细自然语言 reasoning profile，只描述推理路径和方法，不评价质量，不使用答案正确性、投票结果或其他 agents 信息。
4. 根据主/子策略加权分布计算 team diversity、family homogeneity、个体重复度和 reward。
5. 窗口达到 `update_every` 后，group critic 基于策略摘要和 family 分布生成诊断。
6. rewriter 为高压力 agents 生成候选 prompt。
7. bandit 在 `keep_current + candidates` 中采样动作，并用小批样本比较 reward 后决定采纳或拒绝。
8. 写入 step 日志、update 日志、trace 日志、reasoning summary 日志和 prompt 历史。

Group critic、textual gradient 和 rewriter 的输入只包含策略摘要、family 分布、同质化统计、trace hash 与 agent 角色信息；不传入 gold answer、各 agent 的预测答案、vote answer 或 vote correctness。答案和投票结果只用于 reward 计算、评估指标与回溯日志。

## 输出文件

默认输出目录为 `runs_tg_rl/`，也可通过 `--out_dir` 指定。

- `run_meta.json`：配置、初始化方式、初始 prompt hash、模型参数。
- `history.json`：每个 epoch 的 train/test 汇总指标。
- `prompt_history.json`：每个 agent 的 prompt 初始化、更新、采纳、拒绝和 sanitize 事件。
- `update_logs.jsonl`：prompt 更新尝试日志。
- `train_step_logs.jsonl`：每个训练 step 的 family 指标、reward 摘要和 update 摘要。
- `train_trace_history.jsonl`：训练阶段完整推理轨迹。
- `test_trace_history.jsonl`：测试阶段完整推理轨迹。
- `reasoning_summary_history.jsonl`：按样本保存每个 agent 的 family、family resolution、详细 `reasoning_summary`、`summary_embedding_text`、证据片段、confidence 和 trace hash。
- `family_taxonomy.json`：动态 family taxonomy。
- `test_epoch*_predictions.jsonl`：测试样本的答案、投票结果和 family 指标。
- `last_state.json` / `best_state.json`：agent 状态、bandit 参数和 prompt 历史快照。

## 实验脚本

默认四个设置：

- `shared_div`：shared 初始化，训练并开启 diversity reward。
- `bank_div`：bank 初始化，训练并开启 diversity reward。
- `shared_baseline`：shared 初始化，只测试不训练。
- `bank_baseline`：bank 初始化，只测试不训练。

运行实验：

```powershell
python scripts/run_experiments.py --workspace . --out_root runs_experiments --task_type auto --train_path mmlu_train.jsonl --test_path mmlu_test.jsonl --epochs 2 --agents 5
```

读取已有结果并画图：

```powershell
python scripts/analyze_experiments.py --workspace . --out_root runs_experiments
```

也可以拆开执行：

```powershell
python scripts/compute_experiment_metrics.py --runs_root runs_experiments --out_csv runs_experiments/experiment_metrics.csv --out_md runs_experiments/experiment_metrics.md
python scripts/plot_experiment_results.py --csv runs_experiments/experiment_metrics.csv --out_dir runs_experiments/figures
python scripts/plot_experiment_dynamics.py --base_dir runs_experiments --out_dir runs_experiments/figures
python scripts/plot_training_comparison.py --csv runs_experiments/experiment_metrics.csv --out_dir runs_experiments/figures
```

结果分析默认使用 `BAAI/bge-small-en-v1.5` 计算 `summary_embedding_text` 的 embedding cosine similarity/diversity；如需跳过可加 `--disable_summary_embedding`。

方法公式与抽象流程见 [method.md](method.md)。
