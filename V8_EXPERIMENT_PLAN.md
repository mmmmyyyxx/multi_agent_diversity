# V8 Experiment Plan

## Isolation

Run from branch `v8-competence-depth` and write to `runs_vote_v8_competence_depth_<commit>`. Never reuse v7 formal or pilot roots.

## Stage 1: Offline Tests

```powershell
python -m pytest -q
```

This stage uses zero API calls.

## Stage 2: Strict Smoke

Use `disambiguation_qa`, seed 42, five agents, one epoch, at least 20 train/validation/test examples, beam 3, and two candidates per parent. Run the matched v7 baseline, strict legacy, schedule-only, depth2, and full progressive settings.

Require zero split overlap, zero v8 truncated prompts, depth1/oracle agreement, no-tie depth3/vote agreement, complete checkpoint state, and non-empty candidate beams.

## Stage 3: Matched Pilot

Run `disambiguation_qa` and `sports_understanding`, seeds 42 and 43, with the same five settings and matched budgets.

## Stage 4: Decision

Advance only when most comparisons show higher bottom-2 accuracy and C2, non-decreasing C3/vote, lower aggregation gap, lower rescue concentration, no truncation, and no candidate starvation. Oracle need not increase.

## Offline Audit

```powershell
python scripts/analyze_competence_depth.py <run_root>
```

The script writes five CSV files and `COMPETENCE_DEPTH_AUDIT.md` without API calls.
