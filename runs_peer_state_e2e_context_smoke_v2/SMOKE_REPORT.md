# Real API Role Transport And Context Smoke Report

Run date: 2026-07-23

Code identity:

```text
commit: ddbd076f2c844313c0bf76606e4535837947a530
method: peer_state_counterfactual_v1
git_dirty: false
task: disambiguation_qa
seed: 42
```

## Scope

The smoke was intentionally limited to transport, execution integrity, cache
reuse, budgets, and context-isolation auditing. It was not intended to estimate
method quality.

The role transport smoke called Solver, Teacher, Critic, and Student
independently. The end-to-end smoke ran:

- `shared_independent_accuracy_tcs` (B1)
- `shared_peer_state_responsibility` (B3)
- `shared_peer_state_full` (B4)

Each end-to-end run used one epoch, one update, 8/8/8 train/validation/test
examples, one candidate request, a Stage B budget of one, concurrency one, and
resume disabled.

## Role Transport

Status: **passed**

| Check | Result |
|---|---:|
| Solver output valid | yes |
| Teacher schema valid | yes |
| Critic schema valid | yes |
| Student schema valid | yes |
| Student candidates | 1 |
| API calls | 4 |
| Tokens | 2,707 |
| Failed attempts | 0 |

Critic rejection did not fail this smoke because Student transport is tested
independently from the method approval gate.

## End-To-End Results

| Setting | Vote acc. | Mean agent acc. | Valid rate | Calls | Tokens | Shared cache hit/miss | Critic approvals | Stage A/B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 independent accuracy | 0.500 | 0.500 | 1.000 | 30 | 24,720 | 0/24 | 0/3 | 0/0 |
| B3 responsibility | 0.500 | 0.500 | 1.000 | 6 | 12,601 | 24/0 | 0/3 | 0/0 |
| B4 full | 0.500 | 0.500 | 1.000 | 6 | 13,393 | 24/0 | 0/3 | 0/0 |

All calls succeeded. The three runs stayed below their per-run limits of 300
calls and 400,000 tokens. B3 and B4 reused all 24 frozen Solver observations
created by B1, and all runs recorded the same shared-cache content hash:

```text
77226806fafeb53581b3ad76e2d0602c17f36f1558ca0542c12524c7af1fc7ff
```

## Context Isolation

| Setting | Context class | Forbidden-field violations | Responsibility-specific fields |
|---|---|---:|---:|
| B1 | `AccuracyProposalContext` | 0 | 0 |
| B3 | `PeerStateProposalContext` | 0 | 0 |
| B4 | `ResponsibilityProposalContext` | 0 | 21 |

The context audit therefore matches the B1/B3/B4 isolation protocol.

## Acceptance Result

Infrastructure checks passed:

- all three API roles were reachable;
- Solver, Teacher, Critic, and Student schemas were valid;
- all final Solver outputs were valid;
- shared prompt-question caching behaved as intended;
- call and token budgets were enforced and auditable;
- final, cost, identity, TCS-round, context, and candidate artifacts were
  written for all runs;
- the runner exited normally with an empty stderr stream.

The end-to-end candidate-search criterion did **not** pass. All nine Critic
decisions rejected their Teacher proposals, so the method gate correctly
prevented end-to-end Student calls. Consequently no candidate entered Stage A
or Stage B.

This isolates the remaining failure to Teacher-proposal and Critic semantic
quality, rather than endpoint transport, JSON parsing, Solver validity, cache
reuse, budgets, or context isolation. Common rejection themes were vague or
non-executable repair procedures, weak preservation rules, and inaccurate
interpretation of residual or peer-state evidence.
