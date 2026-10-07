# Dataset-informed manual prompt capacity probe

Completed one valid independent development diagnostic: six prompts, two fresh realizations each, Optimize60 only; 720 logical Solver evaluations. Exact manual texts and training examples remain private. All six prompt identities, the fixed P1-P5 team, evaluation order, source and finite budget were frozen before real responses.

Frozen source: `498d667586e8a29de7dffe38f8f50542c8fb0f3c`. Model: qwen3-8b, thinking disabled, temperature 0.2, unchanged benchmark interface and first-valid Solver invalid recovery. No Gradient, Cluster, Reflection, Optimizer or judge calls. No A4 prompt, Memory, Pattern, Responsibility, team or gate changes. Original attempt6 evidence remains immutable.

| Prompt | Replicate 1 correct /60 | Replicate 2 correct /60 | Mean /60 | Correct range |
|---|---:|---:|---:|---|
| P0_BASELINE | 22 | 22 | 22 | 22–22 |
| P1_GENERALIST | 23 | 23 | 23 | 23–23 |
| P2_SPECIALIST_A | 25 | 25 | 25 | 25–25 |
| P3_SPECIALIST_B | 17 | 17 | 17 | 17–17 |
| P4_SPECIALIST_C | 18 | 18 | 18 | 18–18 |
| P5_SPECIALIST_D | 21 | 23 | 22 | 21–23 |

| Fixed manual team | Vote /60 | Oracle /60 | Mean member accuracy | Min member accuracy | Disagreement | Novel Oracle coverage | Novel coverage without correct Vote |
|---|---:|---:|---:|---:|---:|---:|---:|
| replicate1 | 21 | 27 | 0.3467 | 0.2833 | 0.3241 | 5 | 3 |
| replicate2 | 23 | 27 | 0.3533 | 0.2833 | 0.3397 | 5 | 1 |

**Q6 — generalist capacity:** P1 mean is 23/60 versus fresh P0 mean 22/60 and the unchanged original 22/60 reference. P1 beats the original reference: True; beats fresh P0: True. Both comparisons are descriptive and in sample.

**Q7 — specialist coverage:** novel coverage observed: True; highest specialist realization novel count is 5, versus the original A4 Full best of four. Retention, loss and invalidity are reported for every realization, without rebasing U0.

**Q8 — fixed team:** plurality improvement above 22/60 is OBSERVED in one realization. Fixed-team Vote is 21/60 and 23/60, averaging 22/60; this is not a consistent gain across the two realizations. The main five prompts are unchanged and selected before results; no best-five selection is substituted.

**Q9 — optimization space:** single-prompt space `OBSERVED`, coverage space `OBSERVED`, team Vote space `OBSERVED`; A4 search gap `INCONCLUSIVE` and plurality-conversion bottleneck `SUPPORTED` under the frozen descriptive rules. Best manual single realization is 25/60; best manual mean is 25/60.

Fresh P0 is a stochastic reference, not a replacement for the original A4 evidence. Prompt design used Optimize data, so none of these results establishes held-out generalization, method causality or optimality. Two realizations provide means and ranges; no p-values are reported. A4 remains unchanged and is not restarted. Unfavorable valid outcomes are accepted.

P2 preserves all 22 original correct examples and adds three in each realization. P5 obtains five novel examples in its second realization, exceeding the historical single Full best of four, while still losing four original correct examples. The fixed teams retain 19 original team-correct examples: novel coverage converts to two correct votes in replicate 1 and four in replicate 2, against three old team-correct losses in each. These observations support the recorded capacity and conversion classifications without establishing an optimizer cause.

Provider execution: 720 logical evaluations, 819 successful physical calls, zero transport failures and cache hits; 32 terminal invalid logical results and 3 recovered logical results. Frozen recovery was applied only to invalid outputs; valid wrong outputs were accepted without resampling.

Accounting: this diagnostic charged 342,089 tokens; cumulative 3,414,772 of 40M, remaining 36,585,228, outstanding reservation zero. Every transport realization has a durable sealed response/error receipt and reconciled charge. The finite 30M attempt ceiling and cumulative journal were preserved.

Verification: 1538 current tests passed, 2 skipped; 1468 historical/private cases explicitly deselected. Full historical replay passing is not claimed. Compileall, current registry-v2 audit, manifest validation, source/old-report/parent-raw hashes and diff checks pass. The older governance CLI also fails on untouched starting metadata; that baseline limitation is recorded in engineering_verification.json.

Shadow, Validation and Test model calls are zero. No held-out raw samples were parsed in this diagnostic. The consumed authorization is closed. Exact raw outputs, prompt texts and source questions remain under ignored storage; public reports contain hashes, categories, counters and metrics.

Evidence: [frozen manifest](manual_prompt_capacity_probe_manifest.json), [redacted design](manual_prompt_design.json), [all single results](single_prompt_results.csv), [fixed teams](manual_team_results.json), [one-member replacements](replacement_analysis.csv), [Pareto](manual_prompt_pareto.csv), [manual/A4 comparison](manual_vs_a4_full_candidates.csv), [classification](capacity_probe_summary.json), [integrity](integrity_audit.json), [accounting](accounting_audit.json), [engineering checks](engineering_verification.json).
