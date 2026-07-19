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
| Candidate refill controller | `optimization/prompt_update_controller.py`: `CandidateClassificationAndRefillStage` | `maybe_update_prompts` -> `PromptUpdateMixin.update_prompt_with_beam` | `candidate_refill_*` | update summary, starvation counters | `tests/test_v8_search_loop.py`; TCS tests |
| Cheap prescreen | `multi_dataset_diverse_rl/search_archive.py`: `cheap_prescreen` | initial and refill candidate paths | prompt hard limit, required candidate types | rejection summary | `test_cheap_prescreen_rejects_duplicate_and_incomplete_candidates`, `test_mechanism_alternative_requires_observed_operation_change` |
| Safe/Probation/Catastrophic classification | `search_archive.py`: `candidate_quality_bucket` | candidate evaluation and refill completion | probation and catastrophic thresholds | `archive_bucket` | `test_candidate_bucket_distinguishes_safe_probation_and_catastrophic`, novelty tests |
| Safe QD archive | `search_archive.py`: `select_safe_archive` | `update_prompt_with_beam` | `qd_archive_size_per_agent` | `AgentState.safe_qd_archive` | long archive and over-capacity tests |
| Probation archive TTL | `persistence/runtime_state.py`: `_expire_agent_probation_branches` | parent selection and epoch-end cleanup | `probation_archive_ttl_updates` | `AgentState.probation_archive`, expiry counter | `test_expired_probation_branch_is_removed_before_parent_selection` |
| QD parent selection | `search_archive.py`: `select_reproduction_parent`; `persistence/runtime_state.py`: `_select_stable_qd_parents` | each selected agent update | `qd_niche_min_parent_opportunities_per_epoch`, `probation_parent_enabled` | per-niche counts, probation parent count | probation and Safe round-robin tests |
| Mechanism normalization | `multi_dataset_diverse_rl/mechanisms.py`: `normalize_mechanism_representation` | `_attach_stable_mechanism_representation` | mechanism policy version | canonical operations/filtered residual text | `tests/test_mechanism_distance.py` |
| Semantic mechanism families | `mechanisms.py`: specificity and family helpers; `persistence/runtime_state.py`: attachment/cache | candidate/archive/probe profiling | semantic merge and specificity thresholds | family representatives/mappings, embedding cache | `tests/test_semantic_mechanisms.py` |
| Behavior distance | `multi_dataset_diverse_rl/behavior_profiles.py`: `behavior_distance` | QD archive and joint selector | four behavior weights and shrinkage | combination behavior profiles | same-wrong and no-support tests |
| Wrong-answer dispersion | `behavior_profiles.py`: `behavior_distance` | pairwise combination scoring | `behavior_wrong_answer_dispersion_weight` | diagnostic component | `test_same_wrong_dispersion_rewards_different_wrong_answers` |
| Prompt static profile cache | `behavior_profiles.py`: `build_prompt_static_profile` | `_evaluate_prompt_on_stable_probe` | none | answer/correctness/invalid only | team-dependent rescue test and cache assertion |
| Team combination metrics | `quality_diversity.py`: `team_quality_metrics`, `team_diversity_metrics` | `enumerate_joint_teams` | formal plurality/tie-break/parser | team profiles rebuilt per combination | 243 enumeration and team-dependent rescue tests |
| Joint active-team selection | `quality_diversity.py`: `select_stable_joint_team`; `qd/joint_controller.py`: `select_joint_active_team` | epoch-end Stable-QD path | joint quality, change-limit, lineage settings | joint history/latest metrics | joint selection tests |
| Real quality-anchor frontier | `qd/quality_anchors.py`: `update_quality_anchor_archive`, feasibility | joint selection before quality bands | integer loss tolerances, capacity 5 | actual prompt hashes and `QualityCounts` | `tests/test_quality_anchor_archive.py` |
| Hierarchical epsilon bands | `quality_diversity.py`: `hierarchical_quality_bands` | before diversity in `select_stable_joint_team` | integer quality floors and band counts | named per-layer counts | hierarchical band and fold rejection tests |
| Two-fold stability | `quality_diversity.py`: `deterministic_probe_folds`, fold scoring in `select_stable_joint_team` | joint selector | seed offset, exactly two folds | fold metrics/gaps | deterministic folds, fold quality, single-fold lineage tests |
| Active-change limit | `quality_diversity.py`: `active_prompt_change_count`; `persistence/runtime_state.py`: `_current_joint_change_limit` | team enumeration filter | early/late limits and one-epoch relaxation | rejection count and patience state | prompt-hash change test |
| Lineage state machine | `multi_dataset_diverse_rl/lineage.py`: `update_lineage_state`, `lineage_drift` | selected team commit/switch path | two-snapshot commit/switch | `AgentState.lineage_state` | two-snapshot, fold gate, checkpoint tests |
| Peer collapse | `quality_diversity.py`: peer penalties/hard rejection in `select_stable_joint_team` | final-band team scoring | soft/hard similarity thresholds | rejection and penalty counters | near-duplicate peer test |
| QD readiness residual floor | `optimization/lifecycle.py`: `_recompute_effective_residual_strength` | joint selection and competence schedule update | QD readiness and floor | raw/effective strengths | residual-floor recomputation test |
| Agent target fairness | `metrics/rollout_metrics.py`: `select_reward_agents_for_update` | `maybe_update_prompts` | fairness flag and minimum updates | per-agent update counts | three Stable-QD fairness tests |
| Checkpoint serialization | `persistence/checkpoint.py`: build, migrate, validate, restore | training boundaries/resume | generated behavior fingerprint | archives, semantic families, real anchors, caches, lineages | checkpoint v6 round-trip and v5 migration/refusal tests |
| Run metadata export | `persistence/runtime_state.py`: `write_run_meta` | system initialization | canonical and flat `Config` views | `run_meta.json` | setting CLI propagation and smoke validation |
| History output | `persistence/artifacts.py` and `artifact_methods.py`; `cli.py` lifecycle records | updates, probes, joint selection, final evaluation | versioned artifact schemas | JSON/JSONL histories | artifact and smoke validation |

## Policy Ordering And Invariants

1. Candidates pass cheap prescreen before solver evaluation.
2. Safe, Probation, and Catastrophic are exclusive archive outcomes. Probation never enters joint representatives.
3. Bounded refill receives structured prior failures and checks raw, retained-archive, and representative requirements before stopping.
4. The long-term Safe archive is six items; active, highest-quality, and behavior-complementary representatives are selected from all archive candidates, with only three per agent entering offline team enumeration.
5. Prompt-level cache stores intrinsic outcomes only. Rescue, unique-correct, shared-error, same-wrong, coverage, plurality, and behavior distance are rebuilt for each team.
6. Active-change filtering and integer quality floors run before hierarchical bands and diversity scoring. A candidate must pass the current incumbent's local floor and at least one actual team in the real quality-anchor frontier. If none pass, the incumbent fallback is explicit.
7. Two deterministic folds supply quality guards and `mean(diversity) - 0.5 * gap` stability. Lineage evidence uses peer-relative correctness residual and rescue/unique-correct support across folds, not raw fold accuracy gap.
8. A first stable snapshot is provisional; the second can commit. Committed anchors alone create drift and peer-collapse pressure.
9. `competence_schedule_strength` remains the raw schedule. QD readiness may raise only `effective_residual_strength`.

## Compatibility Notes

- `quality_frontier_count` is a legacy output name for the final hierarchical-band count. `quality_floor_feasible_count`, named band counts, and `final_candidate_team_count` are the authoritative Stable-QD diagnostics.
- `epsilon_quality_frontier` remains available for historical tests/settings but is not called by `select_stable_joint_team`.
- Historical lineage epoch names remain flat compatibility aliases; Stable-QD uses `lineage_commit_required_snapshots=2` and `lineage_switch_confirmation_snapshots=2` in the lineage section.
- Checkpoint v6 is current. Version 5 Stable-QD state migrates only when historical joint-selection records contain actual selected prompt hashes and metrics.

## Acceptance Commands

```powershell
$PY = "D:\Anaconda\envs\DL\python.exe"
& $PY -m pytest -q
git diff --check
```

Only the single targeted `disambiguation_qa`, seed 42, two-epoch acceptance smoke specified by the acceptance task may be launched after local checks pass. No formal experiment is part of this audit.

## Post-Refactor Smoke Result

`runs_v8_stable_qd_refactor_smoke_3196576` completed the one permitted smoke:

| Check | Result |
| --- | --- |
| Scope | `disambiguation_qa`, seed 42, 2 epochs, 20/20/20 |
| Split | strict manifest, zero cross-split question overlap |
| TCS | 8/8 update summaries complete; 22 final Student candidates; 0 parse failures |
| Archive/refill | one post-archive refill; trigger was retained niche/task-repair collapse; 2 rounds |
| Representatives | 3 per agent; up to 3 distinct representative niches |
| Mechanisms | 4 observed canonical family IDs plus unknown incumbent; no natural semantic fallback candidate |
| Joint selection | 243 theoretical teams in epoch 2; 163 evaluated after change limit; 73 feasible |
| Real anchors | 1 non-dominated actual team after epoch 2; 73 anchor-feasible teams; no fallback |
| Lineage | 2 uncommitted, 2 provisional, 1 committed |
| Final metrics | vote 0.35, mean individual 0.39, oracle 0.55 |
| Calls | 1,188 solver, 46 optimizer, 30 evaluator; 1,264 total |
| Completion | final history/results present; transient checkpoint cleared; runner exited |

The natural smoke did not emit a residual-only semantic family. Specificity,
semantic merge/separation, stable family IDs, and checkpoint persistence are
covered deterministically in `tests/test_semantic_mechanisms.py` and
`tests/test_training_checkpoint_resume.py`.
