# Strict v2 Cache-Chain Static Audit

- Gate: **PASS**
- BLOCKER: 0
- MAJOR: 0
- Starting commit: `6b1882ed521dd0601e972db59de1636c5b6cc0f3`
- Method: `member_aware_peer_state_v8`
- Checkpoint: `16`

The initial review found two reliability defects in the pre-run audit path:
same-key merge conflicts were silently ignored, and the comparison artifact did
not prove per-question unchanged-prompt equality or account for missing
reference entries and provider recalls. The repair is limited to request/cache
identity, immutable merge checks, and sanitized audit evidence. Responsibility,
target scheduling, proposal context, Stage A/B selection, candidate acceptance,
budgets, catch-up, and Proposal Memory semantics are unchanged.

After repair, the cumulative reference is content-addressed, every local clone
is independently validated, same-key conflicts retain the first observation and
fail the gate, and final-test observations are compared by question hash,
observation hash, parsed-answer hash, correctness, invalid status, and team-vote
vector hash.
