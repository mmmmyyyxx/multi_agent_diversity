# V8 Stable QD Acceptance Audit

This audit covers the existing setting `shared_vote_tcs_competence_depth2_progressive_residual_hybrid`. The setting name is unchanged; its active `method_version` is `v8_stable_qd_lineage`.

## Active Version Contract

| Field | Active value |
| --- | --- |
| `method_version` | `v8_stable_qd_lineage` |
| `target_selector_version` | `hybrid_competence_boundary_v2` |
| `beam_policy_version` | `quality_diversity_archive_v1` |
| `active_team_selector_version` | `joint_quality_diversity_v1` |
| `lineage_policy_version` | `stable_lineage_anchor_v1` |
| `mechanism_distance_version` | `mechanism_sequence_embedding_v1` |
| `candidate_refill_version` | `quality_feedback_refill_v1` |
| `archive_policy_version` | `safe_probation_qd_archive_v1` |
| `joint_quality_filter_version` | `hierarchical_epsilon_band_v1` |
| `probe_stability_version` | `deterministic_two_fold_v1` |
| `parent_selection_version` | `active_plus_round_robin_niche_v1` |

`safe_exploit_explore_v1`, the epsilon-Pareto helper, and three-epoch lineage fields remain only for historical settings or output compatibility. They are not called by the Stable-QD active policy.

## Implementation Map

| Requirement | File and implementation | Active call entry | Configuration | Persistent state | Behavioral tests |
| --- | --- | --- | --- | --- | --- |
| Candidate refill controller | `multi_dataset_diverse_rl/system.py`: `TraceBeamSearchSystem.update_prompt_with_beam` refill loop | `maybe_update_prompts` -> `update_prompt_with_beam` | `candidate_refill_*` | update summary, starvation counters | `tests/test_v8_search_loop.py`; TCS tests |
| Cheap prescreen | `multi_dataset_diverse_rl/search_archive.py`: `cheap_prescreen` | initial and refill candidate paths | prompt hard limit, required candidate types | rejection summary | `test_cheap_prescreen_rejects_duplicate_and_incomplete_candidates`, `test_mechanism_alternative_requires_observed_operation_change` |
| Safe/Probation/Catastrophic classification | `search_archive.py`: `candidate_quality_bucket` | candidate evaluation and refill completion | probation and catastrophic thresholds | `archive_bucket` | `test_candidate_bucket_distinguishes_safe_probation_and_catastrophic`, novelty tests |
| Safe QD archive | `search_archive.py`: `select_safe_archive` | `update_prompt_with_beam` | `qd_archive_size_per_agent` | `AgentState.safe_qd_archive` | long archive and over-capacity tests |
| Probation archive TTL | `system.py`: `_expire_agent_probation_branches` | parent selection and epoch-end cleanup | `probation_archive_ttl_updates` | `AgentState.probation_archive`, expiry counter | `test_expired_probation_branch_is_removed_before_parent_selection` |
| QD parent selection | `search_archive.py`: `select_reproduction_parent`; `system.py`: `_select_stable_qd_parents` | each selected agent update | `qd_niche_min_parent_opportunities_per_epoch`, `probation_parent_enabled` | per-niche counts, probation parent count | probation and Safe round-robin tests |
| Mechanism normalization | `multi_dataset_diverse_rl/mechanisms.py`: `normalize_mechanism_representation` | `_attach_stable_mechanism_representation` | mechanism policy version | canonical operations/filtered residual text | `tests/test_mechanism_distance.py` |
| Mechanism embedding cache | `system.py`: `_attach_stable_mechanism_representation` | candidate/archive/probe profiling | embedding model and distance weights | `mechanism_embedding_cache`, hit/miss counters | `test_stable_probe_and_mechanism_caches_report_real_hits` |
| Behavior distance | `multi_dataset_diverse_rl/behavior_profiles.py`: `behavior_distance` | QD archive and joint selector | four behavior weights and shrinkage | combination behavior profiles | same-wrong and no-support tests |
| Wrong-answer dispersion | `behavior_profiles.py`: `behavior_distance` | pairwise combination scoring | `behavior_wrong_answer_dispersion_weight` | diagnostic component | `test_same_wrong_dispersion_rewards_different_wrong_answers` |
| Prompt static profile cache | `behavior_profiles.py`: `build_prompt_static_profile` | `_evaluate_prompt_on_stable_probe` | none | answer/correctness/invalid only | team-dependent rescue test and cache assertion |
| Team combination metrics | `quality_diversity.py`: `team_quality_metrics`, `team_diversity_metrics` | `enumerate_joint_teams` | formal plurality/tie-break/parser | team profiles rebuilt per combination | 243 enumeration and team-dependent rescue tests |
| Joint active-team selection | `quality_diversity.py`: `select_stable_joint_team`; `system.py`: `select_joint_active_team` | epoch-end Stable-QD path | joint quality, change-limit, lineage settings | joint history/latest metrics | joint selection tests |
| Hierarchical epsilon bands | `quality_diversity.py`: `hierarchical_quality_bands` | before diversity in `select_stable_joint_team` | integer quality floors and band counts | named per-layer counts | hierarchical band and fold rejection tests |
| Two-fold stability | `quality_diversity.py`: `deterministic_probe_folds`, fold scoring in `select_stable_joint_team` | joint selector | seed offset, exactly two folds | fold metrics/gaps | deterministic folds, fold quality, single-fold lineage tests |
| Active-change limit | `quality_diversity.py`: `active_prompt_change_count`; `system.py`: `_current_joint_change_limit` | team enumeration filter | early/late limits and one-epoch relaxation | rejection count and patience state | prompt-hash change test |
| Lineage state machine | `multi_dataset_diverse_rl/lineage.py`: `update_lineage_state`, `lineage_drift` | selected team commit/switch path | two-snapshot commit/switch | `AgentState.lineage_state` | two-snapshot, fold gate, checkpoint tests |
| Peer collapse | `quality_diversity.py`: peer penalties/hard rejection in `select_stable_joint_team` | final-band team scoring | soft/hard similarity thresholds | rejection and penalty counters | near-duplicate peer test |
| QD readiness residual floor | `system.py`: `_recompute_effective_residual_strength` | joint selection and competence schedule update | QD readiness and floor | raw/effective strengths | residual-floor recomputation test |
| Agent target fairness | `system.py`: `select_reward_agents_for_update` | `maybe_update_prompts` | fairness flag and minimum updates | per-agent update counts | three Stable-QD fairness tests |
| Checkpoint serialization | `multi_dataset_diverse_rl/cli.py`: `build_training_checkpoint`, `restore_system_state`, fingerprint checks | training boundaries/resume | `BEHAVIOR_CONFIG_FIELDS` | all archives, counters, caches, lineages | checkpoint round-trip and old-V8 rejection tests |
| Run metadata export | `system.py`: `_base_log_fields`, `write_run_meta` | system initialization | full `Config` | `run_meta.json` | setting CLI propagation and smoke validation |
| History output | `system.py`: `_flush_jsonl`, `save_state`; `cli.py`: epoch/final writers | updates, probes, joint selection, final evaluation | output directory | JSON/JSONL histories | smoke output validation |

## Policy Ordering And Invariants

1. Candidates pass cheap prescreen before solver evaluation.
2. Safe, Probation, and Catastrophic are exclusive archive outcomes. Probation never enters joint representatives.
3. Bounded refill receives structured prior failures and stops on requirements, round limit, unique-candidate limit, optimizer failure, or no new unique candidate.
4. The long-term Safe archive is six items; only three representatives per agent enter offline team enumeration.
5. Prompt-level cache stores intrinsic outcomes only. Rescue, unique-correct, shared-error, same-wrong, coverage, plurality, and behavior distance are rebuilt for each team.
6. Active-change filtering and run-level integer quality floors run before hierarchical bands and diversity scoring. The component-wise anchor combines the initial fixed probe, historical best, committed lineage anchors, and current incumbent, preventing cumulative per-epoch loss.
7. Two deterministic folds supply quality guards and `mean(diversity) - 0.5 * gap` stability. Lineage evidence uses peer-relative correctness residual and rescue/unique-correct support across folds, not raw fold accuracy gap.
8. A first stable snapshot is provisional; the second can commit. Committed anchors alone create drift and peer-collapse pressure.
9. `competence_schedule_strength` remains the raw schedule. QD readiness may raise only `effective_residual_strength`.

## Compatibility Notes

- `quality_frontier_count` is a legacy output name for the final hierarchical-band count. `quality_floor_feasible_count`, named band counts, and `final_candidate_team_count` are the authoritative Stable-QD diagnostics.
- `epsilon_quality_frontier` remains available for historical tests/settings but is not called by `select_stable_joint_team`.
- Historical lineage epoch fields remain in `Config`; Stable-QD uses `lineage_commit_required_snapshots=2` and `lineage_switch_confirmation_snapshots=2`.

## Acceptance Commands

```powershell
$PY = "D:\Anaconda\envs\DL\python.exe"
& $PY -m pytest -q
git diff --check
```

Only the single targeted `disambiguation_qa`, seed 42, two-epoch acceptance smoke specified by the acceptance task may be launched after local checks pass. No formal experiment is part of this audit.
