# Canonical Version And Preset Map

The public setting names remain stable. Internally, settings resolve through
`ExperimentPreset(base, overrides)` and the strategy registry instead of a
large optional-field record.

| Version | Canonical setting | Main strategy composition | Retained formal run | Checkpoint support |
| --- | --- | --- | --- | --- |
| Baseline | `shared_baseline` | shared prompt, no evolution, plurality evaluation | `runs_bbh_oracle_pareto_formal_v2` | Current non-QD v5 can migrate to v6 |
| V1 reward redesign | `shared_guarded_beam` | one-shot generation, scalar guarded-diversity ranking | `runs_task_level_bbh_selected_phase_adaptive` | Current non-QD v5 can migrate to v6 |
| V2 TCS | `shared_guarded_beam` with TCS arguments in its recorded run | Teacher-Critic-Student generation, scalar selection | `runs_task_level_bbh_tcs_useful_full` | Current non-QD v5 can migrate to v6 |
| V3 Oracle Pareto | `shared_oracle_pareto_tcs` | TCS generation, candidate-level oracle Pareto selection | `runs_bbh_oracle_pareto_formal_v2` | Current non-QD v5 can migrate to v6 |
| V7 vote-oriented | `shared_vote_error_pareto_tcs_residual_cycle_guard` | TCS, vote-error Pareto selection, residual and cycle guards | Missing complete formal run; compact pilot records are in `run_records/` | Current non-QD v5 can migrate to v6 |
| V8 Stable-QD | `shared_vote_tcs_competence_depth2_progressive_residual_hybrid` | TCS, Safe/Probation QD archive, behavior-aware representatives, real-team anchors, joint selector, lineage policy | Missing formal run; latest smoke is `runs_v8_stable_qd_refactor_smoke_3196576` | v6; v5 requires real selected-team evidence |

The V8 setting name intentionally still contains historical terms. Its
authoritative identity is `method_version=v8_stable_qd_lineage`.

## Preset Resolution

`scripts/experiment_config.py` defines reusable bases and sparse overrides.
Every override is checked against the canonical field registry. Unknown fields
and type-invalid values fail before a run starts. Flat CLI names remain
compatibility aliases that map to one of the 11 configuration sections.

Historical run directories are interpreted from their own `run_meta.json`,
not from the current meaning of a setting name.
