# Prompt Evolution Audit

## 1. Data coverage and missing fields

- Runs discovered: 1
- Evaluated candidates: 221
- Optimizer candidates: 144
- Full five-distance reconstruction coverage: 0.041
- Proposal-mechanism metadata coverage: 0.339
- Completed runs / checkpoint-bearing runs: 1 / 0
- Parent distance, guard status, Pareto status, outcome, accept-count sequence, and warmup/support conditions are reconstructable.
- Full candidate text and proposal modified-mechanism fields are absent for many rejected/non-retained candidates; their derived lengths and non-parent distances are left blank rather than guessed.

## 2. Candidate prompt-distance distribution

Pilot-only statistics (the partial formal run is excluded):
- Mean / p50 / p75 / p90 parent distance: 0.2343 / 0.1851 / 0.2767 / 0.4918
- P(change_ratio > 0.45 | optimizer candidate): 0.1111
- Evidence supports excessive full-rewrite jumps: **False**

## 3. Active prompt-distance distribution

- P(change_ratio > 0.45 | active top-1 optimizer candidate): 0.0714
- Active-before distance coverage: 14 / 24 (0.5833)
- Known active-before mean / p50 / p75 / p90: 0.2140 / 0.1931 / 0.2596 / 0.3491
- Active prompt update rate: 0.5952
- Agent-level cumulative path and direct distance are reported separately in `prompt_agent_path_summary.csv`; blank path segments indicate unavailable full prompt text, not zero movement.

## 4. Large-shift candidate analysis

- Large optimizer candidates: 16
- P(trust rejection | change_ratio > 0.45): 0.1875
- Warmup exemptions and five support predicates are reconstructed from chronological active updates, run config, and logged candidate metrics.

## 5. Trust-region rejection decomposition

- Unsupported-large-shift rejections: 3
- Trust rejection / optimizer candidate: 0.0208
- Individual rejected candidates and failed support predicates are available in `prompt_candidate_trajectory.csv`.

## 6. Full candidate-selection funnel

Candidate outcomes are mutually exclusive and separate original guards, cycle/trust guards, Pareto retention, existing-beam wins, and active top-1 selection. Evaluated candidates cannot represent pre-evaluation schema/redundancy-filtered proposals; attempt-level raw/final counts preserve that funnel stage.

Pilot optimizer outcomes: `{"accuracy_guard_rejected": 59, "active_top1": 24, "dependence_guard_rejected": 9, "exact_prompt_cycle_rejected": 1, "pareto_not_retained": 19, "retained_nonactive_beam": 29, "unsupported_large_prompt_shift": 3}`

## 7. Residual vs Cycle/Trust paired comparison

- Matched task x seed pairs: 0
- See `prompt_guard_paired_comparison.csv`; active-update differences are decomposed into underfill, dependence, cycle, trust, Pareto non-retention, and existing-beam wins.

## 8. Agent cumulative evolution paths

`prompt_agent_path_summary.csv` distinguishes cumulative adjacent movement from direct distance to the initial prompt, exposing both gradual drift and return-toward-origin paths.

## 9. Mechanism-contract analysis

- Proposal metadata coverage is 0.339; this is insufficient for an all-candidate semantic mechanism analysis.
- Large-shift proposal metadata coverage: 1 / 16 (0.0625).
- Explicit modified-mechanism coverage among large shifts: 0 / 16; present rate within that known subset: NA.
- Explicit preserved-mechanism coverage among large shifts: 1 / 16; missing-list rate within that known subset: 0.0000.
- Small-text/large-behavior cases (distance <0.20 and transition L1 >0.25): 64.
- Repeated target-family mentions: 0 / 0 (NA).
- No new LLM labeling was used. Preserved-mechanism counts are reported only where archived state metadata can be matched by prompt hash.

## 10. Interpretation of evolution speed

Evolution-speed conclusions use active update rate, active step distance, cumulative path, direct initial distance, and logged behavior deltas. Missing full prompt bodies are explicitly separated from genuine zero-distance updates.

## 11. Whether current trust region is too strict

- Trust-region-too-strict criterion (>20% optimizer rejection): **False**.
- Observed pilot rate: 0.0208.

## 12. Whether patch-based generation needs a separate v8 experiment

The offline thresholds do not justify changing v7 mid-experiment. Patch-based generation may still be tested later as a separate v8 ablation, using a new commit and output root.
