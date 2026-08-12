# Multi-Agent Diversity

This repository implements Repairability-Adjusted Dual-Target Prompt-Team
Optimization:

```text
method_version = member_aware_peer_state_v15
checkpoint_version = 25
```

C2/C3 boundary-repair plus preservation contexts are currently explicit
experimental Module2 variants. They do not replace the canonical v15 method or
the four-setting main ablation matrix. C0/C2/C3 carry distinct run and
checkpoint identities.

Five prompts are jointly optimized for equal-weight plurality voting. Each
candidate replaces only one member; dual-target search evaluates two
independent single-member branches from the same parent state and commits at
most one winner.

## Method overview

```text
Joint G/H/M diagnosis
  -> counterfactual (DeltaV, DeltaM) eligibility
  -> unique service routing and one active lane per member
  -> W1 repairability-adjusted target score
  -> Top-1 or Top-2 target branches
  -> responsibility-conditioned Teacher-Critic-Student evolution
  -> fixed-peer common-safe Stage A/B evaluation
  -> one global branch winner
  -> one atomic prompt-team transition
```

The final method has two core modules:

1. Repairability-Adjusted Member-Aware Dual-Target Search
2. Responsibility-Conditioned Evolution

Common-Safe Team Update is the shared candidate write-back rule, not a third
core contribution.

For each actionable member:

```text
B = 0.5 Dhat + 0.3 Shat_support + 0.2 dhat
rho = 1 / (1 + state-local branch failures)
score = (B + 0.05 wait_hat) * rho
```

The wait term is inside the repairability discount. All normalization is
recomputed over the current actionable set. v15 has no target Pareto frontier,
freeze/unfreeze mechanism, persistent reputation, Beta realizability, or
cross-state failure counter.

Every optimized main setting uses target/vote non-regression, strict
target-or-vote progress, terminal-invalid non-regression, and common monotone
ranking. Lane-only progress is not sufficient for acceptance.

## Main settings

| Setting | Display name | Module vector |
|---|---|---:|
| `shared_static_reference` | Static Reference | outside vector; no optimization |
| `shared_generic_evolution` | S0 Generic Prompt Evolution | `00` |
| `shared_member_aware_dual_target` | S1 Member-Aware Dual-Target Search | `10` |
| `shared_responsibility_conditioned_dual_target` | S2 Responsibility-Conditioned Evolution (Full) | `11` |

Static uses zero branches, S0 uses `1 x 2`, and S1/S2 use `2 x 2`. Consequently
S0-to-S1 is the effect of Module 1 as implemented, not a compute-matched causal
estimate. Explicit auxiliary compute controls remain available with
`--allow_auxiliary_setting 1`.

The former v14 `shared_full_dual_target_rcru` setting is no longer a main
setting. It requires `--allow_legacy_setting 1` and retains its historical RCRU
acceptance/ranking semantics for replay and offline analysis. Historical v11,
v12, and v13 settings remain explicit legacy controls as well.

## Offline verification

Use the project conda environment:

```powershell
$PY = "D:\Anaconda\envs\DL\python.exe"

& $PY -m compileall -q multi_dataset_diverse_rl scripts tests
& $PY -m pytest -q
& $PY scripts\preflight_member_aware.py --workspace . --allow_dirty 1
& $PY scripts\deterministic_repairability_selector_smoke.py
& $PY scripts\deterministic_dual_target_competition_smoke.py
& $PY scripts\deterministic_v15_protocol_smoke.py
& $PY scripts\deterministic_final_state_protocol_smoke.py
git diff --check
```

These commands are offline and do not authorize real API calls.

## Running experiments

Run preflight before a separately authorized API experiment:

```powershell
& $PY scripts\run_task_level_accuracy.py `
  --workspace . `
  --manifest configs\task_level_comparison_strict_bbh_seed42.yaml `
  --tasks disambiguation_qa `
  --settings shared_static_reference,shared_responsibility_conditioned_dual_target `
  --seeds 46 `
  --dataset_format mars `
  --out_root experiments\runs_v15_seed46
```

The active lifecycle uses no validation selection. The final active team is
tested once after training only. `solver_max_tokens=1800` remains fixed.
Checkpoint resume accepts only checkpoint v25 with exact v15 run identity and
experimental Module2 context identity. The version bump persists the minimal
accepted-state stable-correct intersection required for exact experimental P3.

## Key artifacts

v15 main runs retain the hash/counter-based artifacts:

- `repairability_adjusted_target_scores.jsonl`
- `dual_target_branch_decisions.jsonl`
- `dual_target_commit_decisions.jsonl`
- `repairability_failure_events.jsonl`
- `repairability_reset_events.jsonl`
- `tcs_context_history.jsonl`
- `candidate_decisions.jsonl`

RCRU-specific artifacts are not required for v15 main runs. Sanitized artifacts
must not contain prompt text, questions, gold/model answers, raw model output,
credentials, endpoints, cache contents, checkpoints, or absolute local paths.

See [method.md](method.md) for the formal algorithm and [AGENTS.md](AGENTS.md)
for normative engineering guardrails.

## v16 diagnostic workflows

The responsibility-coherence audit and the fixed-parent Generic-vs-M20 probe
are noncanonical diagnostics. They do not change Module 1, the S0/S1/S2
matrix, or Common-Safe Team Update. The offline audit is run with
`scripts/audit_v16_responsibility_coherence.py`; the API probe must first pass
`scripts/preflight_v16_generic_m20_probe.py` and a clean source freeze produced
by `scripts/freeze_v16_generic_m20_probe.py`. Only the exact preregistered probe
may use `GENERIC_M20_FIXED_PARENT_PROBE_AUTHORIZED=1`; neither workflow reads
validation or test data or commits a candidate prompt.
