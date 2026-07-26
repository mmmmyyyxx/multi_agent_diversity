# v4 Baseline + Full Pilot (seed 42)

This directory contains the sanitized evidence bundle for the real-model
`disambiguation_qa` pilot at commit `689a5c5cb4320720dd712a31889afd1eb2dcc219`.
The run used `shared_baseline` followed by `shared_member_aware_full`, with
`gpt-4o-mini` for Solver, Teacher, Critic, and Student.

## Gate result

The Full run completed all 8 updates without infrastructure terminal failure.
Five updates were accepted and three were rejected. Invalid Student responses
were recovered by same-cycle retry three times; no upstream Teacher-Critic
regeneration was needed. All valid candidates entering Stage A were retained;
raw invalid responses did not enter Stage A.

Validation evaluated 6 unique team states, reused 3 cached states, and selected
the final checkpoint. Test was evaluated once after selection and was not used
for selection.

The aggregate v4 behavior was observed directly: one accepted candidate had
sample-level vote/pivotal losses while improving the formal aggregate objective.

The operational gate did **not** pass:

```text
eventual Solver validity = 98.727%  (required >= 99.5%)
terminal Solver invalids = 28        (required <= 1)
```

Therefore this run is not a valid formal efficacy result and Independent must
not be run yet. The selected-test signal was positive but remains provisional:

```text
Baseline vote = 52/125
Full vote     = 64/125
member gains  = [+9, +21, +2, +9, +21]
```

## Contents

- `full_gate_analysis.json`: compact gate decision, target sequence, accepted
  candidate metrics, recovery counts, and Baseline-vs-Full comparison.
- `full_final_summary.json`, `baseline_final_summary.json`: sanitized test and
  selection summaries without per-question rows.
- `candidate_decisions_sanitized.jsonl`: candidate funnel, aggregate objective,
  gain/loss, feasibility, and rejection evidence; prompt text removed.
- `candidate_funnel.json`, `history.json`: update-level operational history.
- `student_recovery_observations_sanitized.jsonl`: Student retry audit.
- `target_priority_audit_sanitized.jsonl`,
  `responsibility_assignments_sanitized.jsonl`: scheduling and responsibility
  audit with question identifiers removed from values.
- `solver_recovery_summary.json`, `cost_summary.json`,
  `llm_calls_sanitized.jsonl`: aggregate validity, token, call, retry, and
  latency accounting.
- `run_meta_sanitized.json`: versioned protocol/config metadata without paths,
  endpoints, prompts, or cache identity.
- `sha256_manifest.json`: SHA-256 hashes for this sanitized bundle.

Raw prompts, raw model responses, response excerpts, questions, answers,
SQLite cache, checkpoints, endpoint values, and local absolute paths are not
included.
