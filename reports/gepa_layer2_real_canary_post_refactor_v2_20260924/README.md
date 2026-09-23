# Post-refactor GEPA Layer-2 real-provider canary v2

Classification: **GEPA_LAYER2_REAL_EMPIRICAL_PATH_CONFIRMED**.

This single-attempt engineering canary completed from the exact frozen execution source. Two changed proposals were observed. One failed the unchanged proposal contract; the other passed, reached candidate Solver evaluation, produced a local empirical comparison, and was accepted as a local mutation. Its local minibatch signal was positive. No candidate survived TeamMiniBatch, and there was no full-team evaluation or commit. The canary therefore confirms the real Layer-2-to-GEPA local empirical path, **not** team-level transfer or method efficacy.

| Measure | Observed |
| --- | ---: |
| Proposal attempts / changed | 2 / 2 |
| Contract-invalid / Solver-reached | 1 / 1 |
| Full local evaluations / accepted local mutations | 1 / 1 |
| TeamMiniBatch survivors / team commits | 0 / 0 |
| Solver / Reflection provider calls | 114 / 2 |
| Successful / failed provider attempts | 116 / 0 |
| Input / output / total tokens | 34,035 / 14,560 / 48,595 |
| Validation50 / Test50 calls | 0 / 0 |

The run lifecycle ended `EXECUTION_COMPLETE` after the one-time authorization was consumed. Ledger totals reconcile with the execution summary and lifecycle; record identities are unique, token arithmetic is consistent, and only frozen canary stages and roles appear. Source files were not changed. The execution worktree passed 73 focused tests, `compileall`, and `git diff --check` after the run.

This report contains only sanitized aggregate evidence. The private run root, prompts, examples, responses, provider details, full ledger, and checkpoints remain ignored and are not published. No further API attempt is authorized by this result.
