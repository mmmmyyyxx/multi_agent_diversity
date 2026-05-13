# 运行命令模板

这些命令是模板。运行时使用你的 DL 环境：

```powershell
D:\conda\envs_dirs\DL\python.exe
```

任何需要调用 LLM 的实验开始前，都要先设置 API 环境变量。

## 通用设置

小规模 pilot 推荐：

```powershell
$PY = "D:\conda\envs_dirs\DL\python.exe"
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
$PY = "D:\conda\envs_dirs\DL\python.exe"
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

P4 跨 LLM：保持 judge 为 `gpt-4o-mini`，替换 solver model：

```powershell
& $PY scripts\run_strategy_probe.py `
  --task_type mmlu `
  --test_path $TEST `
  --test_size 100 `
  --prompts_json prove_experiments\prompts\mixed_strategy_mmlu.json `
  --out_dir "$OUT\P4_mixed_strategy_SECOND_MODEL_seed42" `
  --model YOUR_SECOND_SOLVER_MODEL `
  --critic_model gpt-4o-mini `
  --family_expansion_model gpt-4o-mini `
  --family_expansion_enabled 0 `
  --family_taxonomy_path auto `
  --seed 42
```

P2/P3/P4 汇总：

```powershell
& $PY scripts\analyze_prove_experiments.py `
  --runs_root "$OUT" `
  --out_csv "$OUT\prove_summary.csv" `
  --out_md "$OUT\prove_summary.md"
```

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
  --out_md prove_experiments\runs\prove_summary.md
```
