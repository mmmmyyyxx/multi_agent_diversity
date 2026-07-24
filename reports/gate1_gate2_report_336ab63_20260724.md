# Gate 1 / Gate 2 Report

## Scope

- Code commit: `336ab63f41fe37cb6e24a4fb7f721b8fdcbbcb90`
- Method: `member_aware_peer_state_v2`
- Setting: `shared_member_aware_full`
- Task: `disambiguation_qa` (BBH)
- Seed: `42`
- Solver limit: `solver_max_tokens=1800`
- Gate 1 and Gate 2 used separate fresh output/cache directories.
- No API calls were made while preparing this report.

## Gate 1: Contract Smoke

Artifact directory:

```text
gate1_contract_smoke_336ab63_20260724_221737
```

Configuration:

- 2 targeted historical cases: known length-failure case and known Markdown-format case.
- 8 normal control questions.
- 3 repetitions per case; 30 Solver calls total.
- Formal `solver_system_prompt()` and strict parser.
- Each case/repetition used a separate empty SQLite cache.

Result:

| Check | Result |
|---|---:|
| Records | 30 / 30 |
| Valid outputs | 30 |
| `missing_final_answer` | 0 |
| `multiple_final_answers` | 0 |
| `unparseable_final_answer` | 0 |
| `out_of_domain_answer` | 0 |
| `finish_reason=length` | 0 |
| Exactly one final-answer line | 30 / 30 |
| Gate 1 | PASS |

The previously observed long-case truncation and Markdown-wrapped final answer
did not reproduce under the frozen output interface.

## Gate 2: Per-Update Validation

Artifact directory:

```text
runs_gate2_per_update_validation_336ab63_20260724
```

Configuration:

- Optimization / validation / test split sizes: `75 / 50 / 125`.
- `epochs=8`, `update_every=75` (one update opportunity per epoch).
- `candidate_eval_pool_size=75`, `stage_b_candidate_budget=2`.
- Fresh shared Solver cache with 1,900 ready entries.
- Strict split integrity: optimization-validation, optimization-test, and validation-test overlap were all zero.

Validation trajectory:

| Epoch | Feasible | Vote correct | Member correct counts | Invalid rate | Member gains |
|---:|:---:|---:|---|---:|---|
| 1 | yes | 29 | `[29, 29, 41, 29, 29]` | 0.000 | `[0, 0, 12, 0, 0]` |
| 2 | yes | 29 | `[29, 38, 41, 29, 29]` | 0.000 | `[0, 9, 12, 0, 0]` |
| 3 | no | 32 | `[29, 38, 41, 29, 28]` | 0.000 | `[0, 9, 12, 0, -1]` |
| 4 | no | 34 | `[34, 38, 41, 29, 28]` | 0.000 | `[5, 9, 12, 0, -1]` |
| 5-8 | no | 34 | `[34, 38, 41, 29, 28]` | 0.000 | `[5, 9, 12, 0, -1]` |

Validation selected `epoch=2`, the best feasible nonzero checkpoint. It improved
two members without regression and retained the initial validation vote count.
Later epochs were correctly excluded because member 4 fell below its initial
competence floor.

## Selected Test Team

| Metric | Initial | Validation-selected | Delta |
|---|---:|---:|---:|
| Plurality vote correct | 49 / 125 | 50 / 125 | +1 |
| Plurality vote accuracy | 0.392 | 0.400 | +0.008 |
| Per-member correct counts | `[49, 49, 49, 49, 49]` | `[49, 103, 96, 49, 49]` | `[0, +54, +47, 0, 0]` |
| Mean member accuracy | 0.392 | 0.5536 | +0.1616 |
| Minimum member correct gain | 0 | 0 | 0 |
| Improved members | 0 | 2 | 2 |
| Regressed members | 0 | 0 | 0 |
| Selected invalid rate | 0.016 | 0.0128 | -0.0032 |

## Operational Funnel And Cost

- 8 update opportunities; 4 accepted optimization updates.
- 16 Stage A/B candidates; 16 reached Stage B.
- No Teacher/Critic/Student transport or schema terminal failures.
- 1,900 Solver calls; 16 optimizer calls; 8 evaluator calls.
- 1,924 successful LLM calls; 0 failed attempts.

## Conclusion

Gate 1 is fully passed. Gate 2 is operationally successful and produced a
nonzero feasible validation checkpoint (`selected_epoch=2`) with zero validation
invalid outputs. The selected test team shows a small vote improvement and
substantial gains for two members without member regression.

This is not yet a matched baseline-versus-full efficacy result: only the Full
setting was run in Gate 2. A matched efficacy pilot remains necessary for a
causal comparison against `shared_baseline`.

## Reproducibility Note

The Gate 2 artifact records the required code commit and frozen Solver request
template, but its historical `run_identity.git_dirty` field is `true` because
the standalone Gate 1 script and local smoke artifacts were present during the
run. Raw API responses, SQLite caches, and full LLM logs remain local and are
not included in this report.
