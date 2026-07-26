# v4 Independent Accuracy Analysis (seed 42)

This directory contains the sanitized analysis bundle for the v4
`shared_independent_accuracy` run on `disambiguation_qa`. The run used commit
`689a5c5cb4320720dd712a31889afd1eb2dcc219`, five `gpt-4o-mini` agents, 8
updates, and the same matched Baseline root/cache as the Full pilot.

## Run summary

- 8 updates completed; 6 accepted and 2 rejected.
- Target sequence: `0,1,2,3,4,0,1,2`.
- Accepted target sequence: `0,1,2,3,4,0`.
- All five members were successfully targeted at least once.
- Student made 14 calls; 6 invalid responses recovered by same-cycle retry;
  no upstream Teacher-Critic regeneration was triggered.
- Raw invalid Student candidates did not enter Stage A.
- Validation evaluated 7 unique states, reused 2 cached states, and selected
  the final checkpoint.
- Test was evaluated once after selection and was not used for selection.

## Gate diagnostics

The legacy all-request diagnostic remains failed:

```text
eventual validity = 99.36%
terminal invalids = 12
legacy_all_request_validity_gate_v1 = FAIL
```

The 12 terminal invalids came from one rejected exploratory candidate in update
0. Accepted candidates, active incumbent states, validation states, and the
selected test state had zero terminal invalids. Under the provenance-aware
criteria used for the Full amendment:

```text
active_state_validity_and_isolation_v2 = PASS
```

This classification is provided for analysis; it does not alter the formal
Independent protocol or candidate selector.

## Matched test signal

Relative to the shared Baseline test reference:

```text
Baseline vote = 52/125
Independent vote = 52/125
member gains = [+23, 0, 0, 0, 0]
```

The result is therefore useful as the independent-accuracy ablation record but
does not show a team-vote gain in this pilot.

## Contents and privacy

- `independent_analysis.json`: compact update, provenance, gate, recovery, and
  Baseline comparison metrics.
- `independent_final_summary.json`, `baseline_final_summary.json`: sanitized
  summaries without per-question rows.
- `candidate_decisions_sanitized.jsonl`, `candidate_funnel.json`, `history.json`:
  candidate and update-level evidence.
- `student_recovery_observations_sanitized.jsonl`,
  `solver_recovery_summary.json`, `llm_calls_sanitized.jsonl`,
  `cost_summary.json`: recovery and call/token accounting.
- `target_priority_audit_sanitized.jsonl`, `tcs_context_history_sanitized.jsonl`:
  scheduling and context-isolation audit.
- `run_meta_sanitized.json`: protocol/config metadata without paths or cache
  identity.
- `sha256_manifest.json`: SHA-256 manifest for the sanitized files.

Prompts, raw model responses, response excerpts, questions, answers, SQLite
cache, checkpoints, API endpoint values, and local absolute paths are excluded.
The original ignored run directory remains unchanged.
