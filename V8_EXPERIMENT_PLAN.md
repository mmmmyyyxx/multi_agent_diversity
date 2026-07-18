# V8 Experiment Plan

## Isolation

Run from a committed, clean source tree and write to a new commit-tagged output root. Never reuse v7 formal or pilot roots.

## Stage 1: Offline Tests

```powershell
python -m pytest -q
```

This stage uses zero API calls.

## Stage 2: Strict Smoke

Use `disambiguation_qa`, seed 42, five agents, one epoch, at least 20 train/validation/test examples, beam 3, and two candidates per parent. Run the matched v7 baseline, strict legacy, schedule-only, depth2, and full progressive settings.

Require zero split overlap, zero v8 truncated prompts, depth1/oracle agreement,
100% plurality/depth2/reward-component candidate metric coverage, deterministic
tie-break behavior, complete checkpoint state, and non-empty candidate beams.
C3 may differ from plurality vote accuracy.

## Stage 3: Matched Pilot

Run `disambiguation_qa` and `sports_understanding`, seeds 42 and 43, with the same five settings and matched budgets.

## Stage 4: Decision

Advance only when most comparisons show higher bottom-2 accuracy and C2,
non-decreasing actual plurality vote accuracy, a smaller oracle-minus-plurality
gap, lower rescue concentration, improved pivotal-opportunity conversion, no
truncation, and no candidate starvation. C3 remains a redundancy diagnostic,
not a mandatory gate. Oracle need not increase.

## Offline Audit

```powershell
python scripts/analyze_competence_depth.py <run_root>
```

The script writes the original summaries plus schedule, C1-guard, and mechanism summaries, and `COMPETENCE_DEPTH_AUDIT.md`, without API calls. Report aggregation-gap changes together with C1, all-wrong rate, mean accuracy, bottom-2, C2, and vote accuracy; a smaller gap alone is not evidence of improvement.

For V8.2, use `shared_vote_tcs_competence_depth2_progressive_residual_hybrid`. Do not relabel results from the older full-progressive setting as V8.2, and do not treat reused-file `shared_guarded_beam` results as strict same-protocol effect estimates.
