# Full Viability Pilot V1

## Status

Diagnostic only. Do not use this directory as a strict matched baseline/full
comparison because the output root was reused after a logging-only commit:

- baseline run identity: `f13dd7712ec8961550f9e4168c815c68c245f733`
- full run identity: `aac37647bf8033fad5533d86bffece871a2dd043`

Both run identities were clean, but they do not identify the same source
snapshot. The root-level aggregate files were also partially regenerated.

## Final Metrics

| Setting | Vote accuracy | Mean individual accuracy | Minimum individual accuracy | Invalid rate | C0 |
|---|---:|---:|---:|---:|---:|
| Shared baseline | 0.4560 | 0.4560 | 0.4560 | 0.0000 | 68 |
| Peer-state full | 0.6880 | 0.6256 | 0.4560 | 0.0016 | 9 |

The full run had per-agent accuracies:

```text
0.736, 0.672, 0.456, 0.808, 0.456
```

This improvement is a useful viability signal, not a formal matched result.

## Search Funnel

| Metric | Count |
|---|---:|
| Updates | 8 |
| Critic calls | 28 |
| Valid Critic responses | 9 |
| Critic approvals | 6 |
| Raw/schema-valid candidates | 18 / 18 |
| Stage A evaluations | 18 |
| Stage B evaluations | 12 |
| Feasible candidates | 4 |
| Acceptable candidates | 4 |
| Accepted updates | 3 |

Among valid Critic decisions, the approval rate was `6/9 = 66.7%`.

## Schema Diagnosis

| Role | Calls | Schema-valid | Invalid |
|---|---:|---:|---:|
| Teacher | 16 | 15 | 1 |
| Critic | 28 | 9 | 19 |
| Student | 11 | 6 | 5 |

All 19 invalid Critic responses ended at the configured 1,800-token response
limit. All 5 invalid Student responses also ended at 1,800 tokens. The failures
were therefore output truncation, not a reward or candidate-selection failure.

## Guards And Context

- Constraint-feasible Stage B candidates: 4/12.
- Seven candidates were rejected by the invalid-output guard.
- One candidate triggered both vote-loss and pivotal-loss protection.
- Sample-memorizing candidates: 0.
- Forbidden context-field violations: 0.
- Selected target agents: `0, 3, 0, 3, 1, 0, 4, 4`.
- Final responsibility owner distribution: `0:5, 1:6, 2:8, 3:0, 4:8`.

Responsibility did not remain concentrated on one agent.

## Validation And Cost

Validation was feasible:

```text
vote_acc=0.6800
mean_individual_acc=0.6800
invalid_rate=0.0000
```

Full-run cost:

```text
total_llm_calls=1878
successful_llm_calls=1773
failed_llm_attempts=105
total_tokens=1632361
```

The 105 failed attempts were transient rate limits and were recovered.

## Outcome

V1 demonstrated a working search and selection loop, while revealing that the
1,800-token Critic and Student response limits were too small. V2 reran the same
pilot with higher per-response limits and a fresh output root.
