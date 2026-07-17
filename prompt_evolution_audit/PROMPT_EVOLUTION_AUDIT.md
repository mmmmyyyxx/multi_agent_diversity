# Prompt Evolution Audit

## 1. Data coverage and missing fields

- Runs discovered: 21
- Evaluated candidates: 2520
- Optimizer candidates: 1664
- Full five-distance reconstruction coverage: 0.136
- Proposal-mechanism metadata coverage: 0.256
- Completed runs / checkpoint-bearing runs: 20 / 1
- Parent distance, guard status, Pareto status, outcome, accept-count sequence, and warmup/support conditions are reconstructable.
- Full candidate text and proposal modified-mechanism fields are absent for many rejected/non-retained candidates; their derived lengths and non-parent distances are left blank rather than guessed.

## 2. Candidate prompt-distance distribution

Pilot-only statistics (the partial formal run is excluded):
- Mean / p50 / p75 / p90 parent distance: 0.1907 / 0.1617 / 0.2602 / 0.4477
- P(change_ratio > 0.45 | optimizer candidate): 0.0962
- Evidence supports excessive full-rewrite jumps: **False**

## 3. Active prompt-distance distribution

- P(change_ratio > 0.45 | active top-1 optimizer candidate): 0.1453
- Active-before distance coverage: 117 / 216 (0.5417)
- Known active-before mean / p50 / p75 / p90: 0.2768 / 0.2300 / 0.3984 / 0.4838
- Active prompt update rate: 0.6096
- Agent-level cumulative path and direct distance are reported separately in `prompt_agent_path_summary.csv`; blank path segments indicate unavailable full prompt text, not zero movement.

## 4. Large-shift candidate analysis

- Large optimizer candidates: 157
- P(trust rejection | change_ratio > 0.45): 0.0255
- Warmup exemptions and five support predicates are reconstructed from chronological active updates, run config, and logged candidate metrics.

## 5. Trust-region rejection decomposition

- Unsupported-large-shift rejections: 4
- Trust rejection / optimizer candidate: 0.0025
- Individual rejected candidates and failed support predicates are available in `prompt_candidate_trajectory.csv`.

## 6. Full candidate-selection funnel

Candidate outcomes are mutually exclusive and separate original guards, cycle/trust guards, Pareto retention, existing-beam wins, and active top-1 selection. Evaluated candidates cannot represent pre-evaluation schema/redundancy-filtered proposals; attempt-level raw/final counts preserve that funnel stage.

Pilot optimizer outcomes: `{"accuracy_guard_rejected": 680, "active_top1": 216, "dependence_guard_rejected": 101, "pareto_not_retained": 280, "retained_nonactive_beam": 351, "unsupported_large_prompt_shift": 4}`

## 7. Residual vs Cycle/Trust paired comparison

- Matched task x seed pairs: 4
- See `prompt_guard_paired_comparison.csv`; active-update differences are decomposed into underfill, dependence, cycle, trust, Pareto non-retention, and existing-beam wins.

## 8. Agent cumulative evolution paths

`prompt_agent_path_summary.csv` distinguishes cumulative adjacent movement from direct distance to the initial prompt, exposing both gradual drift and return-toward-origin paths.

## 9. Mechanism-contract analysis

- Proposal metadata coverage is 0.256; this is insufficient for an all-candidate semantic mechanism analysis.
- Large-shift proposal metadata coverage: 32 / 161 (0.1988).
- Explicit modified-mechanism coverage among large shifts: 0 / 161; present rate within that known subset: NA.
- Explicit preserved-mechanism coverage among large shifts: 32 / 161; missing-list rate within that known subset: 0.2500.
- Small-text/large-behavior cases (distance <0.20 and transition L1 >0.25): 492.
- Repeated target-family mentions: 0 / 0 (NA).
- No new LLM labeling was used. Preserved-mechanism counts are reported only where archived state metadata can be matched by prompt hash.

## 10. Interpretation of evolution speed

Evolution-speed conclusions use active update rate, active step distance, cumulative path, direct initial distance, and logged behavior deltas. Missing full prompt bodies are explicitly separated from genuine zero-distance updates.

## 11. Whether current trust region is too strict

- Trust-region-too-strict criterion (>20% optimizer rejection): **False**.
- Observed pilot rate: 0.0025.

## 12. Whether patch-based generation needs a separate v8 experiment

The offline thresholds do not justify changing v7 mid-experiment. Patch-based generation may still be tested later as a separate v8 ablation, using a new commit and output root.
