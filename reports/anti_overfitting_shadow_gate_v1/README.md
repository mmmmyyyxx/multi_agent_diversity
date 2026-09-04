# Anti-Overfitting Shadow Gate v1 — Phase A

Phase A freezes an execution-ready, zero-API protocol. Phase B has not run.

- Dataset: DQA 250 rows, repartitioned as TrainDev150 / Validation50 / Test50.
- Cross-fit: three disjoint 50-row folds; each trajectory uses Optimize100 and an unseen Shadow50.
- First causal pilot: `RR_GENERIC_OLD_PROTOCOL` versus `RR_GENERIC_SHADOW_GATED`.
- Solver: `qwen3-8b`; Teacher/Critic/Student/Evaluator: `qwen3.7-flash`.
- Search: all adaptive evidence comes from Optimize100.
- Shadow: exactly one frozen Optimize winner; no ranking, generation, revision, feedback, or retry.
- Write-back: Optimize Common-Safe AND Shadow VoteDelta >= 0 AND target-member loss >= -2/50.
- Budget: at most 32 update opportunities; the Shadow arm stops after six consecutive opportunities without an approved commit.
- Validation50: final frozen state once, after training only.
- Test50: inaccessible in Phase A and the planned pipeline pilot.

The deterministic split used one fixed construction seed and no seed search. Difficulty strata use the already-frozen qwen3-14b D0 predictions from seeds 72/73/74 only; they do not define the new Solver. Across the five 50-row buckets, observed maxima are label TV 0.02, frozen-Static-difficulty TV 0.04, lexical-cluster TV 0.06, and structural |SMD| 0.369.

`PHASE_B_RUN=false`, `API_CALLS=0`, `TEST_CALLS=0`, and `DO_NOT_PUSH=true`.
