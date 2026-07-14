# Stage 2 Selector Pilot Failure Report

## Gate status

Stage 2 did not complete and does not authorize Stage 3. The experiment was
stopped because the model API repeatedly returned a non-transient account
error:

```text
HTTP 429: Access denied, account is not in good standing
error-code#overdue-payment
```

Continuing the configured infinite transient retry loop would only repeat the
same rejected requests. No candidates or tasks were skipped, and partial runs
are not included as completed results.

## Completed run

| Task | Setting | Status | Test vote acc | Mean individual acc | Oracle acc | Solver calls |
|---|---|---|---:|---:|---:|---:|
| disambiguation_qa | shared_baseline | complete | 0.4800 | 0.4736 | 0.5360 | 625 |

The completed run used commit
`83bcb8f90fd4b60344c283b050f5fe4e099a7727`, protocol
`vote_oriented_v3`, and a clean tracked tree.

## Incomplete runs

| Task | Setting | Last durable state | Resume assessment |
|---|---|---|---|
| disambiguation_qa | shared_scalar_tcs_vote_first | training checkpoint, epoch index 0, cursor 9 | Compatible checkpoint exists; resume only after API recovery and full fingerprint validation |
| disambiguation_qa | shared_vote_pareto_tcs | training checkpoint, epoch index 0, cursor 9 | Compatible checkpoint exists; resume only after API recovery and full fingerprint validation |
| geometric_shapes | shared_baseline | partial API logs, no checkpoint/history | Incomplete and not safely classed as resumable from a training step |
| geometric_shapes | shared_scalar_tcs_vote_first | partial API logs, no checkpoint/history | Incomplete and not safely classed as resumable from a training step |

The parent runner was interrupted and all four remaining processes associated
with this output root were explicitly stopped. No Stage 2 process remains.

The following seven planned runs were never started: geometric_shapes Pareto;
all three ruin_names settings; and all three sports_understanding settings.

## Integrity retained

- Existing output files and checkpoints were preserved.
- No partial run was appended to `accuracy_results.jsonl`.
- The two durable checkpoints retain checkpoint version 2, exact behavior
  fingerprints, execution session IDs, prompt state, train order, and cursor.
- Their run metadata records train-only candidate evaluation and zero split
  overlap.
- Stage 1 remains valid and independently reported in
  `runs_vote_stage1_smoke_83bcb8f`.

## Required external action

Restore the configured Aliyun model account to good standing before resuming.
After recovery, resume must use the same Git commit, protocol, models, split
hashes, seed, reward, selector, TCS settings, and candidate-evaluation budget.
