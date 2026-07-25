# Gate 1 / Gate 2 Analysis Report

## Scope

- Code commit: `ad3ba73a89cc579bae906c570a677c72600d955c`
- Method: `member_aware_peer_state_v3`
- Gate 2 setting: `shared_member_aware_full`
- Task: `disambiguation_qa` (BBH/MARS)
- Seed: `42`
- Solver limit: `solver_max_tokens=1800`
- Gate 3 was stopped before completion and is not used in this report.

This report records operational viability only. It is not a matched efficacy
result and does not support a method-performance claim.

## Gate 1: Contract Smoke With Invalid Recovery

Local source artifact:

```text
runs_gate1_invalid_recovery_ad3ba73_20260725_124426
```

The run used 14 directed cases, three repetitions per case, separate fresh
SQLite caches, the production Solver prompt and strict parser. The public
summary is in:

```text
reports/gate_runs/gate1_invalid_recovery_ad3ba73_20260725_124426
```

| Check | Result |
|---|---:|
| Directed cases | 14 |
| Repeated records | 42 |
| Final valid records | 42 / 42 |
| Eventual valid rate | 100% |
| Terminal invalid records | 0 |
| First-pass valid records | 41 / 42 |
| Recovery requests | 1 |
| Recovery API calls | 1 |
| Recovery tokens | 633 |
| Cache replay generated new Solver calls | 0 |
| Recovery summaries reconciled | yes |

The single recovered record was `historical_missing_final_04`. Its first
attempt ended with `finish_reason=length` and `missing_final_answer`; the
request-local recovery returned a valid response with one final-answer line.
Therefore:

- **Recovery viability:** PASS.
- **Strict zero-first-pass-invalid contract:** not fully passed.
- **Interpretation:** the immutable output interface and recovery path work,
  but a rare output-window exhaustion remains an audited runtime event. It
  must not be presented as a zero-truncation result.

The recovery overhead was 2.38% of calls and 2.47% of tokens for the complete
42-record run. Raw responses, per-case SQLite caches, and runner logs remain
local.

## Gate 2: One-Update Full Pipeline

Local source artifact:

```text
runs_gate2_one_update_v3_ad3ba73_20260725
```

Configuration was `epochs=1`, `update_every=75`, split sizes `75/50/125`,
`candidate_eval_pool_size=75`, `stage_b_candidate_budget=2`, a fresh shared
Solver cache, and `solver_invalid_max_retries=3`. The public summary is in:

```text
reports/gate_runs/gate2_one_update_v3_ad3ba73_20260725
```

### Solver and baseline state

- 250 unique Solver requests.
- 249 first-pass valid and 1 recovered invalid.
- 0 terminal invalid requests.
- Eventual Solver validity: 100%.
- Validation baseline: 25/50 vote-correct and 25 correct answers for each
  member; invalid rate 0%.
- No accepted update and no member gain were observed.

### Update funnel

| Stage | Result |
|---|---:|
| Target agents eligible | 5 (`0..4`) |
| Selected target | agent `2` |
| Teacher calls | 2 |
| Critic calls | 2 |
| Critic approvals | 0 |
| Critic semantic rejections | 2 |
| Student calls | 0 |
| Stage A evaluated | 0 |
| Stage B evaluated | 0 |
| Feasible candidates | 0 |
| Accepted candidates | 0 |

The two Critic rejections were schema-valid semantic decisions:

1. `evidence_mismatch`: the first repair plan overclaimed that the supplied
   evidence established a specific antecedent.
2. `actionable_specificity`: the revised plan used a generic “strongly
   supports” criterion without an executable threshold.

The terminal failure was:

```text
critic_semantic_rejection_exhausted
```

This is a TCS proposal-pipeline operational failure. Because no Student
candidate reached Stage A, there is no empirical candidate rejection, Pareto
comparison, validation-selected update, or efficacy conclusion to report.

## Gate Decisions

| Gate | Decision | Reason |
|---|---|---|
| Gate 1 | PASS for recovery viability; conditional for strict first-pass contract | 42/42 eventually valid, with one recovered length failure |
| Gate 2 | FAIL | Critic rejected both semantic rounds; no candidate reached rollout |
| Gate 3 | STOPPED / not evaluated | Process stopped before a valid Gate 2 completion |

## Next Action

Do not start a new Gate 3 or matched efficacy pilot from this run. The next
implementation task should address the real-provider Critic rejection path while
preserving the strict Critic checks and leaving member objectives,
responsibility assignment, Stage A/B, Pareto selection, and the Solver contract
unchanged. Then rerun Gate 2 in a new directory and cache.

## Provenance And Publication Boundary

The public directories contain only secret-free compact summaries. The
following remain local and are intentionally not committed:

- raw API response payloads;
- SQLite Solver caches;
- full `llm_calls` logs and runner logs;
- the incomplete Gate 3 run.
