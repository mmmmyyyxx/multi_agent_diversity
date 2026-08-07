# Multi-Agent Diversity

This repository implements Repairability-Adjusted Dual-Target Prompt-Team
Optimization:

```text
method_version = member_aware_peer_state_v14
checkpoint_version = 23
```

Five prompts are jointly optimized for equal-weight plurality voting. Each
candidate still replaces only one member; dual-target search means evaluating
two independent single-member branches from the same parent state and
committing at most one winner.

## Method overview

```text
Joint G/H/M diagnosis
  -> counterfactual (DeltaV, DeltaM) eligibility
  -> unique service routing and one active lane per member
  -> repairability-adjusted scalar target score
  -> Top-1 or Top-2 target branches
  -> independent Teacher-Critic-Student candidate search
  -> branch-local Stage A/B and safety
  -> one global branch winner
  -> one atomic prompt-team transition
```

For each actionable member:

```text
D = active-lane direct vote flips
S_support = positive margin gain only on non-direct-flip residuals
d = max(0, maximum member gain - member gain - 5)

B = 0.5 Dhat + 0.3 Shat_support + 0.2 dhat
repairability_discount = 1 / (1 + state-local branch failures)
score = B * repairability_discount + 0.05 wait_hat
```

All normalization is recomputed over the current actionable set. v14 does not
use a target Pareto frontier and has no active freeze/unfreeze mechanism.
State-local failure counts reset only after a real accepted team transition.

S0-S2 use the common monotone target-or-vote safety rule. S3 uses branch-local
RCRU. For a role-only candidate (zero vote gain with positive active-lane
progress), v14 Layer 3 requires at least one positive residual, no negative
residual, and a non-negative paired-bootstrap LCB. Cross-branch competition
never commits more than one member and never counts a feasible loser as a
repairability failure.

## Main settings

| Setting | Display name | Added module |
|---|---|---|
| `shared_static_reference` | Static Reference | outside module vector; no optimization |
| `shared_generic_evolution` | S0 Generic Prompt Evolution | vector `000`; generic evolution |
| `shared_member_aware_dual_target` | S1 Member-Aware Dual-Target Search | vector `100`; complete module one |
| `shared_responsibility_conditioned_dual_target` | S2 Responsibility-Conditioned Evolution | vector `110`; compact single-lane context |
| `shared_full_dual_target_rcru` | S3 Robust Contribution Update (Full) | vector `111`; RCRU |

Auxiliary search-budget controls are explicit and excluded from the main
matrix:

```text
aux_dual_target_budget_matched_2x1
aux_single_target_compute_matched_1x4
```

They require `--allow_auxiliary_setting 1`. The former seven-setting v13
protocol and historical v12 names require `--allow_legacy_setting 1` and retain
explicit `legacy_v13_*` or `legacy_v12_*` identities.

## Offline verification

Use the project conda environment:

```powershell
$PY = "D:\Anaconda\envs\DL\python.exe"

& $PY -m compileall -q multi_dataset_diverse_rl scripts tests
& $PY -m pytest -q
& $PY scripts\preflight_member_aware.py --workspace . --allow_dirty 1
& $PY scripts\deterministic_repairability_selector_smoke.py
& $PY scripts\deterministic_dual_target_competition_smoke.py
& $PY scripts\deterministic_v14_protocol_smoke.py
& $PY scripts\deterministic_rcru_smoke.py
& $PY scripts\deterministic_final_state_protocol_smoke.py
git diff --check
```

These commands are offline. They do not require or authorize real API calls.

## Running experiments

Run the preflight before a user-authorized API experiment:

```powershell
& $PY scripts\run_task_level_accuracy.py `
  --workspace . `
  --manifest configs\task_level_comparison_strict_bbh_seed42.yaml `
  --tasks disambiguation_qa `
  --settings shared_static_reference,shared_full_dual_target_rcru `
  --seeds 46 `
  --dataset_format mars `
  --out_root experiments\runs_v14_seed46
```

Static uses no branch or candidate and performs no TCS calls. S0 uses one
target branch and two candidates. S1-S3 use two target branches and two
candidates per branch. Every valid generated candidate enters its branch's
Stage B.

The active lifecycle uses no validation selection. The final active team is
tested once after training only. `solver_max_tokens=1800` remains fixed so
Solver request and shared-cache identity stay stable.

Checkpoint resume accepts only checkpoint v23 with exact v14 run identity.
Older checkpoints fail rather than migrate or silently restart.

## Key artifacts

In addition to the existing final summary, candidate, TCS, cost, and lifecycle
artifacts, v14 writes:

- `repairability_adjusted_target_scores.jsonl`
- `dual_target_branch_decisions.jsonl`
- `dual_target_commit_decisions.jsonl`
- `repairability_failure_events.jsonl`
- `repairability_reset_events.jsonl`

These files are hash/counter based. They must not contain prompt text,
questions, gold/model answers, raw model responses, credentials, endpoints,
cache contents, checkpoints, or absolute local paths.

See [method.md](method.md) for the formal algorithm and [AGENTS.md](AGENTS.md)
for normative engineering guardrails.
