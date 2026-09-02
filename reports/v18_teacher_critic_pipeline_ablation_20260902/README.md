# V18 Teacher-Critic Pipeline Simplification

This fixed-parent four-arm experiment compares canonical Teacher/Critic, Teacher-Clean only, Teacher-Clean without semantic Critic, and Teacher-Clean with a non-blocking advisory Critic. Candidate decisions were frozen before winner-only Val50 evaluation. Validation Vote is the primary architecture selection metric; Oracle is mechanism-only.

Frozen selected arm: **C_NO_SEMANTIC_CRITIC**.

Reason: advisory feedback lacks stable quality benefit.

| Arm | Student reach | Valid | Feasible | WOULD_COMMIT | Val Target delta | Val Vote delta | Val Oracle delta | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A_CANONICAL | 3/6 | 12 | 7 | 3/6 | -5 | +0 | +14 | 336387 |
| B_TEACHER_CLEAN | 2/6 | 6 | 2 | 1/6 | -4 | +0 | +3 | 197805 |
| C_NO_SEMANTIC_CRITIC | 6/6 | 22 | 13 | 5/6 | +0 | +0 | +24 | 550937 |
| D_ADVISORY_CRITIC | 6/6 | 24 | 15 | 5/6 | -11 | +0 | +19 | 599578 |

Teacher-Clean alone reduced pre-Student throughput relative to canonical. Removing the semantic veto restored full Student reach and the strongest validation target transfer. Advisory feedback produced two additional feasible candidates but no additional Vote gain and substantially worse target transfer, so it did not establish a stable quality benefit.

The post-run analyzer reconciles the runtime `valid`/`feasible` candidate keys with the public report schema; this changed no runtime artifact or method decision.

No prompt was committed, no trajectory was mutated, and Test125 was not accessed.

```text
TEST_ACCESSED=false
TEAM_PROMPT_COMMITS=0
```
