# Strict v2 S3/S4/S5 run status

Status: **STOPPED — incomplete matrix, not a formal three-seed comparison**.

The static audit, offline gate, and live no-op witness passed. The frozen source
identity remained unchanged during the formal run. For seed 44, S0, S3, and S4
completed. S5 completed all 32 planned training updates, then its one allowed
final-test evaluation failed with HTTP 403 `local:insufficient_quota`. This is
not a transient timeout or rate-limit error, so the run was not retried and
seeds 45 and 46 were not started.

Only the three complete seed-44 final-test rows are reported as observed
partial evidence. Missing final tests and unstarted runs are explicitly marked
`unavailable`; no historical run was used to fill them. The strict stage gate
is `FAIL` with 3/12 complete runs and nine blockers.

No source/configuration changed after source freezing. No commit or push was
performed.
