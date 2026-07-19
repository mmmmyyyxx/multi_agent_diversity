# V8 Stable QD Lineage Experiment Plan

## Stage 1: Offline Tests

Verify mechanism normalization, behavior distance, Safe/Probation retention, bounded refill feedback, niche parent opportunities, 243-team enumeration, hierarchical quality bands, two-fold stability, lineage transitions, checkpoint rejection, and occupancy bounds.

## Stage 2: Targeted Smoke

Run only `disambiguation_qa`, seed 42, two epochs, and the existing hybrid setting. Confirm:

- `method_version=v8_stable_qd_lineage`;
- early self-drift is zero;
- QD retains incumbent, Safe niches, and bounded Probation branches;
- refill, parent-source, archive, and starvation diagnostics are recorded;
- joint combination, quality-floor, band, and final-selector counts are nonzero;
- active prompts may come from mechanism niches;
- selected mean, bottom-2, C1, and C2 satisfy tolerances;
- behavior and mechanism distances are exported;
- lineage remains valid uncommitted/provisional state;
- checkpoint state restores and old V8 checkpoints fail explicitly;
- no prompt truncation, probe drift, or candidate starvation occurs.
- V8 legacy beam refresh calls are zero;
- skipped joint refreshes preserve active prompts and lineage state;
- dirty shortlist probing is capped and cache hits are reported;
- TCS repair and open mechanism exploration are both routed and logged;
- open exploration and offline team enumeration make no Teacher/Critic or team-level solver calls respectively.

## Stage 3: Matched Pilot

Only after smoke passes, compare matched seeds with identical models, splits, beam size, candidate count, and evaluation budget. Report vote, mean, bottom-2, C1/C2, behavior distance, lineage stability, active source, update funnel, and cost.

## Stage 4: Formal Runs

Formal multi-seed or multi-task experiments require a separate explicit command. A smoke result is an execution-integrity result, not an accuracy claim.

## Historical Results

Old V8 result directories remain readable under their recorded `method_version`. The current setting name has new behavior, so do not merge old and new rows without version filtering.

## Preformal Freeze Checklist

Before the formal three-seed run, the frozen code must pass:

- `362` local pytest tests, `compileall`, and `git diff --check`;
- canonical semantic niche counting and Critic `passed` plus score checks;
- per-question Oracle-to-Vote, C0/C1/C2/C3+, top-tie, and normalization diagnostics;
- deduplicated TCS/Open/incumbent/other candidate funnel accounting persisted
  through checkpoint resume;
- dirty Safe profile coverage and representative distance diagnostics without
  changing shortlist or ranking behavior;
- one completed acceptance smoke in `runs_v8_preformal_acceptance_799df8c`:
  1151 calls, two joint refreshes, zero skipped refreshes, zero legacy refresh
  calls, zero team-level solver calls, and quality constraints passed.

No formal run is launched by this checklist. After freezing, do not resume the
acceptance directory with modified code or behavior arguments.
