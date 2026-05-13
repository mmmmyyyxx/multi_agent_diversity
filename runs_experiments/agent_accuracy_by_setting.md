# Per-Agent Test Accuracy

Accuracy is computed from `test_epoch1_predictions.jsonl` as `answers[agent_id] == gold`.

| setting | vote acc | agent 0 | agent 1 | agent 2 | agent 3 | agent 4 |
|---|---:|---:|---:|---:|---:|---:|
| shared_div | 0.720 (72/100) | 0.710 (71/100) | 0.710 (71/100) | 0.750 (75/100) | 0.760 (76/100) | 0.740 (74/100) |
| bank_div | 0.750 (75/100) | 0.720 (72/100) | 0.700 (70/100) | 0.720 (72/100) | 0.760 (76/100) | 0.750 (75/100) |
| shared_baseline | 0.770 (77/100) | 0.760 (76/100) | 0.750 (75/100) | 0.720 (72/100) | 0.730 (73/100) | 0.750 (75/100) |
| bank_baseline | 0.740 (74/100) | 0.710 (71/100) | 0.740 (74/100) | 0.740 (74/100) | 0.740 (74/100) | 0.750 (75/100) |

## Trace Alignment Check

`test_epoch1_predictions.jsonl` and `test_trace_history.jsonl` currently do not share question hashes, so sampled traces below are shown as independent test-trace references rather than attaching gold/agent correctness to them.

| setting | prediction rows | trace rows | hash intersection |
|---|---:|---:|---:|
| shared_div | 100 | 100 | 0 |
| bank_div | 100 | 100 | 0 |
| shared_baseline | 100 | 100 | 0 |
| bank_baseline | 100 | 100 | 0 |
