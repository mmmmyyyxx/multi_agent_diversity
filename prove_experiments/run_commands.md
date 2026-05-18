# 运行命令模板

这些命令是模板。运行时请把 `$PY` 设置成你当前机器上真实可用、且已安装 `openai` 的 Python 环境。

```powershell
$PY = "你的_python.exe_完整路径"
```

任何需要调用 LLM 的实验开始前，都要先设置 API 环境变量。

## 通用设置

小规模 pilot 推荐：

```powershell
$PY = "你的_python.exe_完整路径"
$TRAIN = "mmlu_train_500.jsonl"
$VAL = "mmlu_val_150.jsonl"
$TEST = "mmlu_test_200.jsonl"
$OUT = "prove_experiments\runs"
```

除非你专门测试 judge 模型敏感性，否则 judge 先统一使用 `gpt-4o-mini`。

## P3 Pilot：显式混合策略 test-only

当前 CLI 支持 `shared` 和 `bank` 初始化。内置 MMLU bank 已经近似混合策略 prompt，因此可以作为第一个低成本干预 pilot：

```powershell
& $PY -m multi_dataset_diverse_rl.cli `
  --task_type mmlu `
  --baseline_only 1 `
  --init_mode bank `
  --test_path $TEST `
  --test_size 100 `
  --agents 5 `
  --model gpt-4o-mini `
  --critic_model gpt-4o-mini `
  --family_expansion_model gpt-4o-mini `
  --family_taxonomy_path auto `
  --use_dual_family_labels 1 `
  --out_dir "$OUT\P3_bank_mixed_gpt4omini_seed42" `
  --seed 42
```

同策略 shared baseline：

```powershell
& $PY -m multi_dataset_diverse_rl.cli `
  --task_type mmlu `
  --baseline_only 1 `
  --init_mode shared `
  --shared_prompt "Use option-by-option elimination. Check each candidate answer against the question, discard inconsistent choices, and output exactly one FINAL_ANSWER line." `
  --test_path $TEST `
  --test_size 100 `
  --agents 5 `
  --model gpt-4o-mini `
  --critic_model gpt-4o-mini `
  --family_expansion_model gpt-4o-mini `
  --family_taxonomy_path auto `
  --use_dual_family_labels 1 `
  --out_dir "$OUT\P2_same_elimination_gpt4omini_seed42" `
  --seed 42
```

## P4 跨 LLM 模板

保持 judge 不变，只替换 solver model，重复 test-only 命令：

```powershell
& $PY -m multi_dataset_diverse_rl.cli `
  --task_type mmlu `
  --baseline_only 1 `
  --init_mode bank `
  --test_path $TEST `
  --test_size 100 `
  --agents 5 `
  --model YOUR_SECOND_SOLVER_MODEL `
  --critic_model gpt-4o-mini `
  --family_expansion_model gpt-4o-mini `
  --family_taxonomy_path auto `
  --use_dual_family_labels 1 `
  --out_dir "$OUT\P4_bank_mixed_second_model_seed42" `
  --seed 42
```

## P5 Reward 权重 sweep

无 diversity reward 的训练对照：

```powershell
& $PY -m multi_dataset_diverse_rl.cli `
  --task_type mmlu `
  --train_path $TRAIN `
  --val_path $VAL `
  --test_path $TEST `
  --train_size 500 `
  --val_size 150 `
  --test_size 200 `
  --agents 5 `
  --init_mode shared `
  --epochs 6 `
  --early_stopping_patience 1 `
  --early_stopping_min_delta 0.005 `
  --candidate_eval_batch_size 10 `
  --update_every 5 `
  --model gpt-4o-mini `
  --critic_model gpt-4o-mini `
  --rewriter_model gpt-4o-mini `
  --family_expansion_model gpt-4o-mini `
  --family_taxonomy_path auto `
  --lambda_diversity 0.0 `
  --lambda_homogeneity 0.0 `
  --lambda_invalid_trace 0.30 `
  --out_dir "$OUT\P5_no_div_seed42" `
  --seed 42
```

默认 diversity reward：

```powershell
& $PY -m multi_dataset_diverse_rl.cli `
  --task_type mmlu `
  --train_path $TRAIN `
  --val_path $VAL `
  --test_path $TEST `
  --train_size 500 `
  --val_size 150 `
  --test_size 200 `
  --agents 5 `
  --init_mode shared `
  --epochs 6 `
  --early_stopping_patience 1 `
  --early_stopping_min_delta 0.005 `
  --candidate_eval_batch_size 10 `
  --update_every 5 `
  --model gpt-4o-mini `
  --critic_model gpt-4o-mini `
  --rewriter_model gpt-4o-mini `
  --family_expansion_model gpt-4o-mini `
  --family_taxonomy_path auto `
  --lambda_diversity 0.5 `
  --lambda_homogeneity 0.35 `
  --lambda_invalid_trace 0.30 `
  --out_dir "$OUT\P5_default_div_seed42" `
  --seed 42
```

softened-tree 条件：

```powershell
& $PY -m multi_dataset_diverse_rl.cli `
  --task_type mmlu `
  --train_path $TRAIN `
  --val_path $VAL `
  --test_path $TEST `
  --train_size 500 `
  --val_size 150 `
  --test_size 200 `
  --agents 5 `
  --init_mode shared `
  --epochs 6 `
  --early_stopping_patience 1 `
  --early_stopping_min_delta 0.005 `
  --candidate_eval_batch_size 10 `
  --update_every 5 `
  --model gpt-4o-mini `
  --critic_model gpt-4o-mini `
  --rewriter_model gpt-4o-mini `
  --family_expansion_model gpt-4o-mini `
  --family_taxonomy_path auto `
  --lambda_diversity 0.5 `
  --lambda_homogeneity 0.35 `
  --lambda_invalid_trace 0.30 `
  --same_major_family_weight 0.7 `
  --out_dir "$OUT\P5_softened_tree_seed42" `
  --seed 42
```

strict-tree 压力测试：

```powershell
& $PY -m multi_dataset_diverse_rl.cli `
  --task_type mmlu `
  --train_path $TRAIN `
  --val_path $VAL `
  --test_path $TEST `
  --train_size 500 `
  --val_size 150 `
  --test_size 200 `
  --agents 5 `
  --init_mode shared `
  --epochs 6 `
  --early_stopping_patience 1 `
  --early_stopping_min_delta 0.005 `
  --candidate_eval_batch_size 10 `
  --update_every 5 `
  --model gpt-4o-mini `
  --critic_model gpt-4o-mini `
  --rewriter_model gpt-4o-mini `
  --family_expansion_model gpt-4o-mini `
  --family_taxonomy_path auto `
  --lambda_diversity 0.5 `
  --lambda_homogeneity 0.35 `
  --lambda_invalid_trace 0.30 `
  --same_major_family_weight 0.25 `
  --out_dir "$OUT\P5_strict_tree_seed42" `
  --seed 42
```

## 汇总结果

```powershell
& $PY scripts\compute_experiment_metrics.py `
  --runs_root "$OUT" `
  --out_csv "$OUT\prove_metrics.csv" `
  --out_md "$OUT\prove_metrics.md"
```

## 关于自定义 per-agent 策略 prompt

当前 CLI 还不能直接传入五个任意 agent 初始 prompt。最强的 P2/P3 证明需要补一个小型 probe runner：从 JSON 读取每个 agent 的 prompt，然后调用现有 evaluation path。补这个 runner 之前，`init_mode bank` 可以做 pilot，但不是最强干预实验。

现在已提供 `scripts/run_strategy_probe.py`，因此推荐优先使用下面的新版命令。

## 新版 P2/P3/P4 自定义 Prompt Probe

先设置变量：

```powershell
$PY = "你的_python.exe_完整路径"
$TEST = "mmlu_test_200.jsonl"
$OUT = "prove_experiments\runs"
```

P2 同策略改写负对照：

```powershell
& $PY scripts\run_strategy_probe.py `
  --task_type mmlu `
  --test_path $TEST `
  --test_size 100 `
  --prompts_json prove_experiments\prompts\same_elimination_mmlu.json `
  --out_dir "$OUT\P2_same_elimination_gpt4omini_seed42" `
  --model gpt-4o-mini `
  --critic_model gpt-4o-mini `
  --family_expansion_model gpt-4o-mini `
  --family_expansion_enabled 0 `
  --family_taxonomy_path auto `
  --seed 42
```

P3 显式混合策略干预：

```powershell
& $PY scripts\run_strategy_probe.py `
  --task_type mmlu `
  --test_path $TEST `
  --test_size 100 `
  --prompts_json prove_experiments\prompts\mixed_strategy_mmlu.json `
  --out_dir "$OUT\P3_mixed_strategy_gpt4omini_seed42" `
  --model gpt-4o-mini `
  --critic_model gpt-4o-mini `
  --family_expansion_model gpt-4o-mini `
  --family_expansion_enabled 0 `
  --family_taxonomy_path auto `
  --seed 42
```

P4 跨 LLM：保持 judge 为 `gpt-4o-mini`，用四个低成本同级 solver model 跑同策略和混合策略矩阵：

```powershell
& $PY scripts\run_p4_cross_llm_matrix.py `
  --workspace . `
  --python $PY `
  --models_json prove_experiments\p4_low_cost_models.json `
  --out_root "$OUT" `
  --test_path $TEST `
  --test_size 100 `
  --critic_model gpt-4o-mini `
  --family_expansion_model gpt-4o-mini `
  --family_expansion_enabled 0 `
  --critic_api_key_env OPENAI_API_KEY `
  --critic_base_url_env OPENAI_BASE_URL `
  --seed 42
```

推荐四个模型在 `prove_experiments\p4_low_cost_models.json` 中配置：

- `gpt-4o-mini`
- `gemini-2.5-flash-lite`
- `qwen2.5-7b-instruct`
- `deepseek-chat`

如果使用统一 OpenAI-compatible 网关，把四个模型的 `solver_api_key_env` 和 `solver_base_url_env` 都改成同一组环境变量即可。如果 Gemini、Qwen、DeepSeek 来自不同供应商，则分别设置：

- `GEMINI_API_KEY`、`GEMINI_OPENAI_BASE_URL`
- `OPENROUTER_API_KEY`、`OPENROUTER_BASE_URL` 或你自己的网关变量
- `OPENAI_API_KEY`、`OPENAI_BASE_URL` 用于固定 GPT-4o-mini critic/judge

先只打印命令、不实际调用 API：

```powershell
& $PY scripts\run_p4_cross_llm_matrix.py `
  --workspace . `
  --python $PY `
  --models_json prove_experiments\p4_low_cost_models.json `
  --out_root "$OUT" `
  --test_path $TEST `
  --test_size 20 `
  --dry_run 1
```

P2/P3/P4 汇总：

```powershell
& $PY scripts\analyze_prove_experiments.py `
  --runs_root "$OUT" `
  --out_csv "$OUT\prove_summary.csv" `
  --out_md "$OUT\prove_summary.md" `
  --out_stats_json "$OUT\prove_stats.json"
```

该汇总现在会额外输出：

- `prove_summary.csv`：run 级指标、P5 candidate 诊断、target hit rate。
- `prove_summary.md`：表格和自动解释提示。
- `prove_stats.json`：P2/P3 question-level paired bootstrap CI、Wilcoxon 近似检验、P4 model identity check。

## P1 Judge 稳定性重判

该命令会从已有 `runs_experiments` 抽样完整 trace，对同一条 trace 重判 3 次：

```powershell
& $PY scripts\rejudge_strategy_traces.py `
  --runs_root runs_experiments `
  --out_dir prove_experiments\rejudge_p1 `
  --max_per_run 25 `
  --repeats 3 `
  --critic_model gpt-4o-mini `
  --family_expansion_model gpt-4o-mini `
  --family_expansion_enabled 0 `
  --family_rejudge_on_low_confidence 0 `
  --family_taxonomy_path auto `
  --seed 42
```

输出：

- `prove_experiments\rejudge_p1\rejudge_summary.md`
- `prove_experiments\rejudge_p1\rejudge_trace_agreement.csv`
- `prove_experiments\rejudge_p1\rejudge_records.jsonl`

## P5 Reward Sweep 新版命令

低成本 pilot 只跑 `no_div/default/softened_tree/strict_tree`：

```powershell
& $PY scripts\run_prove_reward_sweep.py `
  --workspace . `
  --python $PY `
  --out_root prove_experiments\runs `
  --conditions no_div,default,softened_tree,strict_tree `
  --seeds 42 `
  --train_path mmlu_train_500.jsonl `
  --val_path mmlu_val_150.jsonl `
  --test_path mmlu_test_200.jsonl `
  --train_size 500 `
  --val_size 150 `
  --test_size 200 `
  --epochs 6 `
  --early_stopping_patience 1 `
  --candidate_eval_batch_size 10 `
  --update_every 5 `
  --model gpt-4o-mini `
  --critic_model gpt-4o-mini `
  --rewriter_model gpt-4o-mini `
  --family_expansion_model gpt-4o-mini `
  --family_expansion_enabled 0
```

完整 sweep：

```powershell
& $PY scripts\run_prove_reward_sweep.py `
  --workspace . `
  --python $PY `
  --out_root prove_experiments\runs `
  --conditions no_div,weak,default,strong,softened_tree,strict_tree `
  --seeds 42,43,44 `
  --train_path mmlu_train_500.jsonl `
  --val_path mmlu_val_150.jsonl `
  --test_path mmlu_test_200.jsonl `
  --train_size 500 `
  --val_size 150 `
  --test_size 200 `
  --epochs 6 `
  --early_stopping_patience 1 `
  --candidate_eval_batch_size 10 `
  --update_every 5 `
  --model gpt-4o-mini `
  --critic_model gpt-4o-mini `
  --rewriter_model gpt-4o-mini `
  --family_expansion_model gpt-4o-mini `
  --family_expansion_enabled 0
```

P5 跑完后汇总：

```powershell
& $PY scripts\analyze_prove_experiments.py `
  --runs_root prove_experiments\runs `
  --out_csv prove_experiments\runs\prove_summary.csv `
  --out_md prove_experiments\runs\prove_summary.md `
  --out_stats_json prove_experiments\runs\prove_stats.json
```

## P6 Taxonomy 粒度敏感性

该分析不调用 LLM，只复用已有 prediction 文件，离线重算三种粒度：

- `major_only`
- 当前 weighted tree
- `strict_leaf`

```powershell
& $PY scripts\analyze_taxonomy_granularity.py `
  --runs_root prove_experiments\runs `
  --taxonomy_path auto `
  --out_dir prove_experiments\p6_taxonomy
```

如果已经完成 GPT-5.5 盲评，并有 `question_hash,gpt_method_diversity_score` 或 `question_hash,score` 的 CSV/JSONL：

```powershell
& $PY scripts\analyze_taxonomy_granularity.py `
  --runs_root prove_experiments\runs `
  --taxonomy_path auto `
  --blind_annotations prove_experiments\p7_gpt55_blind\p7_gpt55_analysis_rows.csv `
  --out_dir prove_experiments\p6_taxonomy
```

输出：

- `prove_experiments\p6_taxonomy\p6_question_granularity.csv`
- `prove_experiments\p6_taxonomy\p6_granularity_summary.csv`
- `prove_experiments\p6_taxonomy\p6_granularity_summary.md`

## P7 GPT-5.5 盲评代理

先从已有 runs 中抽样 trace group，生成盲评包，然后调用 `gpt-5.5` 做独立盲评。评估器看不到 run、model、prompt、label、gold 或自动指标，只看到匿名 agent trace。

```powershell
& $PY scripts\run_gpt_blind_validation.py `
  --runs_root prove_experiments\runs `
  --out_dir prove_experiments\p7_gpt55_blind `
  --per_bucket 20 `
  --evaluator_model gpt-5.5 `
  --evaluate 1 `
  --seed 42
```

输出：

- `prove_experiments\p7_gpt55_blind\p7_blind_annotation_packet.jsonl`
- `prove_experiments\p7_gpt55_blind\p7_annotation_key.csv`
- `prove_experiments\p7_gpt55_blind\p7_gpt55_evaluations.jsonl`
- `prove_experiments\p7_gpt55_blind\p7_gpt55_analysis_rows.csv`
- `prove_experiments\p7_gpt55_blind\p7_gpt55_analysis.json`
- `prove_experiments\p7_gpt55_blind\p7_gpt55_summary.md`

只生成盲评包、不调用 API：

```powershell
& $PY scripts\run_gpt_blind_validation.py `
  --runs_root prove_experiments\runs `
  --out_dir prove_experiments\p7_gpt55_blind `
  --per_bucket 20 `
  --evaluate 0 `
  --seed 42
```

如果之后想补人工审计，仍可用旧的人工 CSV 回填流程。准备一个 CSV 或 JSONL，至少包含：

- `blinded_id`
- `human_method_diversity_score` 或 `human_method_diversity_score_1_to_5`

然后运行：

```powershell
& $PY scripts\prepare_human_blind_validation.py `
  --runs_root prove_experiments\runs `
  --out_dir prove_experiments\p7_human_blind `
  --per_bucket 20 `
  --annotations prove_experiments\p7_human_blind\completed_annotations.csv `
  --seed 42
```

会额外输出：

- `p7_human_annotation_analysis_rows.csv`
- `p7_human_annotation_analysis.json`

## P8 任务依赖与 Subject-Level 检查

该分析用原始 test jsonl 的 `subject` 字段和 prediction 的 `question_hash` 对齐，不需要重跑模型。

```powershell
& $PY scripts\analyze_task_dependence.py `
  --runs_root prove_experiments\runs `
  --dataset_name mmlu `
  --test_path mmlu_test_200.jsonl `
  --test_size 200 `
  --out_dir prove_experiments\p8_task_dependence
```

输出：

- `prove_experiments\p8_task_dependence\p8_question_rows.csv`
- `prove_experiments\p8_task_dependence\p8_subject_summary.csv`
- `prove_experiments\p8_task_dependence\p8_dataset_summary.csv`
- `prove_experiments\p8_task_dependence\p8_task_dependence_summary.md`

如果你在其他数据集上也跑了 P2/P3 prompt probe，把 `--runs_root`、`--dataset_name` 和 `--test_path` 换成对应数据集即可；最后比较各数据集的 `mean_subject_intervention_effect`。
