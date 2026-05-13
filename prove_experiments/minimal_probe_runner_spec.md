# 最小 Probe Runner 规格

当前 CLI 支持 `init_mode shared` 和 `init_mode bank`，但最强的证明实验需要精确指定每个 agent 的 prompt。因此本规格对应的最小 probe runner 已实现为 `scripts/run_strategy_probe.py`。

## 目的

使用五个显式给定的 agent prompts 跑 test-only 受控实验，同时复用现有 solver、judge、日志和指标计算代码。

## 已实现脚本

`scripts/run_strategy_probe.py`

## 输入参数

- `--task_type mmlu`
- `--test_path mmlu_test_200.jsonl`
- `--test_size 100`
- `--prompts_json prove_experiments/prompts/mixed_strategy_mmlu.json`
- `--out_dir prove_experiments/runs/P3_mixed_strategy_seed42`
- `--model gpt-4o-mini`
- `--critic_model gpt-4o-mini`
- `--family_expansion_model gpt-4o-mini`
- `--family_taxonomy_path auto`
- `--seed 42`

## Prompt JSON 格式

```json
{
  "name": "mixed_strategy_mmlu",
  "agents": [
    {
      "agent_id": 0,
      "target_family": ["concept_definition_match"],
      "prompt": "..."
    },
    {
      "agent_id": 1,
      "target_family": ["distractor_elimination", "option_contrast"],
      "prompt": "..."
    }
  ]
}
```

## 已实现行为

1. 创建 `Config`，其中 `baseline_only=True`，`agents=len(prompts)`，`init_mode=shared`。
2. 实例化 `TextualGradientRLSystem`。
3. 用 JSON 中的 prompt 覆盖 `system.agents[i].current_prompt` 和 `initial_prompt`。
4. 调用 `system.evaluate_dataset(test_data, split_name="test_probe")`。
5. 保存：
   - `run_meta.json`
   - `history.json`
   - `prompt_history.json`
   - `test_probe_predictions.jsonl` 或 `test_epoch1_predictions.jsonl`
   - `test_trace_history.jsonl`
   - `probe_prompts.json`
6. 在 run metadata 中记录：
   - 每个 agent 的 target family。
   - prompt hash。
   - probe name。

## 额外分析

运行后计算：

- instructed-family hit rate。
- same-major hit rate。
- target family 到 judged family 的 confusion table。
- 与同策略负对照相比的 question-level intervention effect。

其中 instructed-family hit rate、same-major hit rate 和 question-level intervention effect 已由 `scripts/analyze_prove_experiments.py` 汇总；更细的 confusion table 可在 `test*_predictions.jsonl` 与 `probe_prompts.json` 上进一步展开。

## 为什么需要这个 runner

`bank` baseline 有参考价值，但它不是干净的实验操纵，因为 bank prompt 不是为证明实验精确设计的。自定义 prompt runner 可以隔离：

- 同一策略、不同措辞。
- 不同策略、相近 prompt 长度。
- 不同模型下完全相同的策略指令。

