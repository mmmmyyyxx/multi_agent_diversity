# Multi-Agent Diversity

This repository implements one current method:
**Member-Aware Prompt-Team Optimization**
(`member_aware_peer_state_v12`).

The system optimizes five solver prompts for equal-weight plurality voting. Model
weights are never updated. Teacher-Critic-Student (TCS) proposes prompt changes,
fixed-probe rollouts evaluate them, and an aggregate non-regression rule plus
the setting-specific selector decides whether a single-agent update enters the
active team.

## Three-Module Method

```text
Current Prompt Team
  -> Joint Team Diagnosis: G, H, M and member gains
  -> Member-Aware Responsibility
     - per-residual lexicographic eligibility on (DeltaV, DeltaM)
     - unique legal-member service routing
     - one anchored active lane and target Pareto on (D, S, d)
  -> Responsibility-Conditioned Evolution
     - one lane, one pattern, <=2 repair cases, <=1 preservation case
  -> Robust Contribution Update
     - target-only paired rollout with four fixed peers
     - monotone safety, active-lane utility, coalition contribution
     - paired support robustness and minimal-edit tie-break
  -> Updated Prompt Team
```

The retained team evaluation objective has three integer-count dimensions:

```text
V_count = correctly aggregated fixed-probe examples
g_i     = candidate correct count for member i - initial correct count for member i
g_min   = min_i g_i
g_sum   = sum_i g_i
```

The common S1-S4 update cannot reduce target correct count or aggregate team
vote, must strictly improve at least one, and cannot increase terminal-invalid
outputs. Those guards imply strict Pareto improvement in
`(V_count, g_min, g_sum)`. S5 preserves target, vote, terminal validity, and
active-lane utility, and permits strict vote or robust active-lane progress.
The method does not require zero loss on every probe example.

Under the same fixed-peer replacement, `g_sum` changes by exactly the target
member's correct-count change. `g_min` distinguishes candidates only when the
selected target is the unique weakest member. The main S1-S4 matrix therefore
does not vary a Member-First key; its broad weak-member mechanism begins in S3
through the uplift-deficit term `d` in target scheduling.

For each vote-wrong residual, only currently wrong members are considered.
Replacing one member by gold while holding four peers fixed gives
`DeltaV` (vote-correct gain) and `DeltaM` (plurality-margin gain). Eligibility
keeps the lexicographic maximum and retains exact ties. These legal portfolios
may overlap. The program routes every serviceable residual to exactly one
eligible, unfrozen member using anchor compatibility, lane load, total load,
and seeded rank. Each member exposes one active repair lane; direct-fix count
`D` and margin-gain sum `S` are aggregated only over that active slice. The
scheduler computes one Pareto frontier over `(D, S, d)`, where `d` is the member
uplift deficit. Wait is only a tie-break inside the first frontier. State-conditioned
repairability freeze excludes a member after two complete failures under the
same portfolio state and returns it only after two accepted updates by other
members plus material portfolio change. There is no waiting-time override or
generic compensation lane.
Only an accepted update sets or switches the target's specialization anchor;
rejection retains it and unfreezing clears it.
`d` is the sole weak-member protection term; an empty-portfolio, frozen, or
strictly dominated weak member receives no additional forced service.

Optional `proposal_memory_mode=state_local_v1` keeps only sanitized failed-search
feedback under a complete run/state/agent/prompt/eligible-residual key. The compact
S5 context never exposes that feedback. It does not change eligibility, scheduling,
objectives, Stage A/B budgets, or evaluation.
Soft vote utility is only a deterministic tie-break signal.

The formal default is `proposal_memory_mode=off`. Proposal Memory is an
optional proposal-search extension, not part of the three-module method.

S1-S4 share the same fixed-peer monotone target-or-vote acceptance,
`common_monotone_safe` ranking, matched candidate budget, and all-generated
Stage A. S2 adds only vote-state diagnosis, S3 adds only responsibility
allocation, and S4 adds only the compact single-lane context. S5 inherits S4's
target, active slice, proposal context, generation, and budgets, and changes
only the candidate decision to RCRU.

## Experiment Settings

Exactly six main settings are supported:

| Setting | Display name | Primary change |
|---|---|---|
| `shared_baseline` | S0 Static Prompt Team | No optimization |
| `shared_generic_evolution` | S1 Generic Prompt Evolution | Generic individual-error proposal scaffold |
| `shared_vote_state_diagnosis` | S2 Vote-State Diagnosis | S1 plus G/H/M and Peer-State diagnosis |
| `shared_member_aware_responsibility` | S3 Member-Aware Responsibility | S2 plus routing, anchors, freeze, and target scheduling |
| `shared_responsibility_conditioned_evolution` | S4 Responsibility-Conditioned Evolution | S3 plus compact single-lane context |
| `shared_full_rcru` | S5 Robust Contribution Update (Full) | S4 plus three-layer RCRU decision |

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
python scripts/deterministic_target_scheduler_smoke.py
python scripts/deterministic_aggregate_acceptance_smoke.py
python scripts/deterministic_rcru_smoke.py
python scripts/deterministic_student_recovery_smoke.py
python scripts/deterministic_final_state_protocol_smoke.py
python scripts/deterministic_high_frequency_update_smoke.py
python scripts/deterministic_team_differentiation_metrics_smoke.py
python scripts/deterministic_proposal_memory_smoke.py
python scripts/historical_proposal_memory_trigger_replay.py
git diff --check
```

The system smoke instantiates the real optimization system with fake models,
runs eight offline fake-model updates through programmatic aggregation, TCS, and
Stage A/B, checks one responsibility
refresh per committed team transition, verifies monotone acceptance and the
derived Pareto invariant, and covers compact eligibility and scheduling. The smaller unit smoke retains deterministic helper-level
coverage. The three v4 smokes separately prove aggregate acceptance, bounded
Student invalid recovery, final-active-state test isolation, planned high-
frequency update counting, and answer-behavior differentiation metrics.

The real-API role transport smoke uses the production Solver limit, omits
completion limits for Teacher/Critic/Student, applies the configured structural
character limits, and accepts any non-empty valid Student candidate set up to
the requested count. Critic calibration records finish reason, provider
truncation, and token usage separately from JSON/schema validity.

## Running Experiments

The default provider credential routing uses the Alibaba Cloud Model Studio
environment variable `DASHSCOPE_API_KEY` for Solver, Optimizer, and Evaluator.
All three active model roles default to the pinned
`qwen3.7-flash-2026-07-15` snapshot. Every request explicitly sends
`enable_thinking=false`; TCS continues to use the existing text-JSON parsing
protocol rather than enabling provider JSON mode.
The default OpenAI-compatible endpoint is:

```text
https://ws-tbeq6fj4ndibcz5p.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
```

Set `DASHSCOPE_BASE_URL` only when an endpoint override is required. API keys
must remain in environment variables and must never be written to configs,
commands, artifacts, or source files. A process started before the Windows
system variable was created must be restarted before it can read the key.

Run the preflight first, then use the task runner:

```powershell
python scripts/run_task_level_accuracy.py `
  --workspace . `
  --manifest configs/task_level_comparison_strict_bbh_seed42.yaml `
  --tasks disambiguation_qa `
  --settings shared_baseline,shared_full_rcru `
  --seeds 42 `
  --dataset_format mars `
  --out_root experiments/runs_member_aware_disambiguation
```

Teacher, Critic, and Student outputs are not truncated by experiment-level completion-token budgets. Their search space is bounded structurally through strict schemas, bounded text fields, a fixed candidate count, and prompt-length constraints. Generic contexts retain their configured bounds; S5 is fixed to one lane, one dominant pattern, at most two repair cases, at most one preservation case, and at most 6000 serialized characters. After a valid Critic rejection, Teacher receives an explicit stateless revision request containing the complete prior plan, structured Critic decision, and the same bounded diagnosis context. A Student response with no valid candidate receives structured error feedback and up to three retries. If all four calls in that cycle are invalid, the program performs one fresh Teacher-Critic regeneration and allows one final four-call Student cycle. A partially valid response is used immediately. Thus one update can make at most eight Student calls, and invalid candidates never enter Stage A. Actual token usage is recorded for post-hoc analysis but does not terminate the experiment.

The Solver retains `solver_max_tokens=1800` so its request identity and shared
cache remain stable. A provider may still return `finish_reason=length`; this
is audited as a runtime failure rather than evidence that the method cannot
improve.

Add explicit sizes, candidate-evaluation budgets, models, and concurrency flags
for a formal run. `--resume_from_checkpoint 1` resumes only an exact
checkpoint-v21 run identity;
incompatible checkpoints fail with an error instead of restarting in place.
`--resume_completed 1` reuses only complete artifacts with an exact identity.

## Main Artifacts

Each optimized run writes:

- `final_summary.json`: final-active-team test metrics and lifecycle summary;
  matched initial-test gains are supplied by the frozen baseline reference
- `best_prompts.json`: final active prompt team
- `history.json`: epoch-level active-probe and funnel summary
- `training_dynamics.jsonl`: initial state and every executed update, including
  rejected-state reuse
- `team_differentiation_trajectory.jsonl` and
  `update_transition_decomposition.jsonl`: answer-behavior geometry and
  accepted-update G/H/M transitions
- `final_test_differentiation.json`: final-team test behavior metrics
- `candidate_decisions.jsonl`: Stage A/B evaluations, guards, and acceptance
- `candidate_funnel.json`: update funnels and role-specific terminal failures
- `responsibility_assignments.jsonl`: counterfactual `(DeltaV, DeltaM)`
  eligibility sets and member portfolios after each refresh
- `target_priority_audit.jsonl`: `(D, S, d)` priorities, frozen and active
  member IDs, the single target Pareto frontier, and selection reason
- `repairability_freeze_events.jsonl` and
  `repairability_unfreeze_events.jsonl`: sanitized state-conditioned budget
  safeguard events
- `service_routing_audit_sanitized.jsonl`: per-state unique routing decisions
  without prompts, questions, answers, or model output
- `specialization_anchor_trajectory_sanitized.jsonl`: anchor initialization,
  accepted set/switch, retained rejection, freeze, and unfreeze-clear events
- `solver_recovery_summary.json`: one row per resolved prompt-question request,
  including first-pass validity, recovery calls, terminal-invalid counts, and
  token overhead
- `tcs_context_history.jsonl` and `tcs_rounds.jsonl`: context isolation and JSON audit
- `student_recovery_observations.jsonl`: retry classes, feedback, upstream
  regeneration, and terminal Student recovery status
- `solver_invalid_outputs.jsonl`: strict `FINAL_ANSWER` failures
- `llm_calls.jsonl` and `cost_summary.json`: role-level API accounting
- `run_meta.json`: frozen method, protocol, cache, and run identity

Candidate prompts optimize only the mutable reasoning procedure. Every Solver
request appends an immutable, task-specific output interface after that
procedure, and the interface explicitly overrides conflicting mutable
instructions. A deterministic validator rejects any mutable prompt containing
the interface marker or copied output instructions at initial load, checkpoint
restore, candidate parsing, accepted-state commit, and Solver construction.
The request-template version participates in Solver request and shared-cache
identity. Hash-only historical audits are available through
`scripts/audit_mutable_prompt_contamination.py`; they do not rewrite artifacts.

Final and task-level summaries report both correct-count gains and normalized
accuracy gains:
`minimum_member_correct_count_gain`, `mean_member_correct_count_gain`,
`minimum_member_accuracy_gain`, and `mean_member_accuracy_gain`. Formal
selection continues to use integer correct counts.

The active final-state evaluation protocol does not run validation during optimization. The
final active team after the fixed update budget or valid repairability early
stop is the selected team; there is
no best epoch, validation cache, rollback, or checkpoint selection. Test runs
exactly once after the optimization lifecycle completes and cannot influence training.
The frozen matched baseline remains a reporting reference only.

See [method.md](method.md) for definitions and implementation details.

Historical scripts whose filenames explicitly name GPT-4o mini keep their
original model pins for reproducibility; they are not active defaults and must
not be mixed into matched Qwen comparisons.

The tracked `reports/v7_frontier_seed46_stage1_20260730` bundle is historical
development evidence for a superseded mechanism. It does not define v12.

For offline analysis of an already completed high-frequency v4 run, use:

```powershell
D:\Anaconda\envs\DL\python.exe scripts\analyze_v4_offline_audit.py `
  --run_dir <local_raw_run_dir> `
  --out_dir <new_sanitized_report_dir>
```

This is an observed-state scheduler decision replay, not a counterfactual
training result. Candidate-search outcome is a realized search result, not an
estimate of member potential. Behavioral subgroups are permitted; only their
error complementarity and stability are evaluated.
