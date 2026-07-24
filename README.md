# Multi-Agent Diversity

This repository implements one current method:
**Member-Aware Peer-State Prompt-Team Optimization**
(`member_aware_peer_state_v2`).

The system optimizes five solver prompts for equal-weight plurality voting. Model
weights are never updated. Teacher-Critic-Student (TCS) proposes prompt changes,
fixed-probe rollouts evaluate them, and a competence-constrained member-aware
Pareto rule decides whether a single-agent update enters the active team.

## Method Flow

```text
Team Rollout
  -> Programmatic Peer-State Pattern Aggregation
  -> Lightweight Repair Hypothesis
  -> Hard-Blocker Critique
  -> Prompt Realization
  -> Stage A: team-vote / worst-member / mean-member shortlist
  -> Pareto Rollout Selection: competence guards + (vote count, minimum gain, total gain)
  -> accepted prompt, then immediate state and responsibility refresh
```

The three formal Stage B objectives are integer counts:

```text
V_count = correctly aggregated fixed-probe examples
g_i     = candidate correct count for member i - initial correct count for member i
g_min   = min_i g_i
g_sum   = sum_i g_i
```

A member-aware candidate is accepted only when its
`(V_count, g_min, g_sum)` vector Pareto-dominates the incumbent. Soft vote
utility is only a deterministic tie-break signal.

## Experiment Settings

Exactly six settings are supported:

| Setting | Purpose |
|---|---|
| `shared_baseline` | Shared prompt, no optimization |
| `shared_independent_accuracy` | Round-robin individual-accuracy ablation |
| `shared_peer_state_vote_first` | Pure vote-first candidate-selection ablation |
| `shared_peer_state_member_pareto` | Adds member-aware Pareto selection |
| `shared_member_aware_responsibility` | Adds member-aware target responsibility |
| `shared_member_aware_full` | Adds member-aware responsibility-conditioned TCS |

All settings use five agents, plurality aggregation, tie-as-abstain, and matched
candidate budgets.

## Offline Verification

No API credentials are needed:

```powershell
python -m pytest -q
python -m compileall -q multi_dataset_diverse_rl scripts
python scripts/preflight_member_aware.py --workspace . --allow_dirty 1
python scripts/deterministic_member_objective_unit_smoke.py
python scripts/deterministic_member_aware_system_smoke.py
# Or run both:
python scripts/deterministic_member_aware_smoke.py
git diff --check
```

The system smoke instantiates the real optimization system with fake models,
runs eight offline fake-model updates through programmatic aggregation, TCS, and
Stage A/B, checks one responsibility
refresh per committed team transition, verifies the two critical Pareto
accept/reject cases, covers all eligible members, and computes the real
validation key. The smaller unit smoke retains deterministic helper-level
coverage.

## Running Experiments

Run the preflight first, then use the task runner:

```powershell
python scripts/run_task_level_accuracy.py `
  --workspace . `
  --manifest configs/task_level_comparison_strict_bbh_seed42.yaml `
  --tasks disambiguation_qa `
  --settings shared_baseline,shared_member_aware_full `
  --seeds 42 `
  --dataset_format mars `
  --out_root runs_member_aware_disambiguation
```

Teacher, Critic, and Student outputs are not truncated by experiment-level completion-token budgets. Their search space is bounded structurally through strict schemas, at most three representative cases, bounded text fields, a fixed candidate count, and prompt-length constraints. Actual token usage is recorded for post-hoc analysis but does not terminate the experiment.

The Solver retains `solver_max_tokens=1800` so its request identity and shared
cache remain stable. A provider may still return `finish_reason=length`; this
is audited as a runtime failure rather than evidence that the method cannot
improve.

Add explicit sizes, candidate-evaluation budgets, models, and concurrency flags
for a formal run. `--resume_from_checkpoint 1` resumes only an exact
checkpoint-v7 run identity;
incompatible checkpoints fail with an error instead of restarting in place.
`--resume_completed 1` reuses only complete artifacts with an exact identity.

## Main Artifacts

Each optimized run writes:

- `final_summary.json`: initial test, selected test, member gains, selection summary
- `best_prompts.json`: validation-selected prompt team
- `history.json`: epoch validation, member objective, and terminal-failure summary
- `candidate_decisions.jsonl`: Stage A/B evaluations, guards, and acceptance
- `candidate_funnel.json`: update funnels and role-specific terminal failures
- `responsibility_assignments.jsonl`: residual ownership after each refresh
- `target_priority_audit.jsonl`: member-aware target priorities and overdue status
- `tcs_context_history.jsonl` and `tcs_rounds.jsonl`: context isolation and JSON audit
- `solver_invalid_outputs.jsonl`: strict `FINAL_ANSWER` failures
- `llm_calls.jsonl` and `cost_summary.json`: role-level API accounting
- `run_meta.json`: frozen method, protocol, cache, and run identity

Final and task-level summaries report both correct-count gains and normalized
accuracy gains:
`minimum_member_correct_count_gain`, `mean_member_correct_count_gain`,
`minimum_member_accuracy_gain`, and `mean_member_accuracy_gain`. Formal
selection continues to use integer correct counts.

See [method.md](method.md) for definitions and implementation details.
