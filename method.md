# Method: Stable Quality-Diversity Lineage Optimization

## 1. Purpose

This repository evolves prompts for a team of solver agents. Model weights are not trained. Reward ranks prompt candidates, while validation selects the state used for final testing.

The current V8 setting is:

```text
shared_vote_tcs_competence_depth2_progressive_residual_hybrid
```

Its current implementation is identified by:

```text
method_version               = v8_stable_qd_lineage
target_selector_version      = hybrid_competence_boundary_v2
beam_policy_version          = quality_diversity_archive_v1
active_team_selector_version = joint_quality_diversity_v1
lineage_policy_version       = stable_lineage_anchor_v1
mechanism_distance_version   = mechanism_sequence_embedding_v1
```

The setting name is retained for experiment continuity, but its old V8.2 behavior has been replaced. Old V8 checkpoints are intentionally incompatible.

## 2. Design Principle

Prompt wording diversity is not the objective. The method seeks agents that remain competent but solve different residual errors on a fixed optimization probe.

The ordering is strict:

1. Hard guards remove invalid or catastrophically degraded candidates.
2. Candidate quality determines the elite within each mechanism niche.
3. Team quality constraints define feasible prompt combinations.
4. Hierarchical integer-count quality bands keep quality-competitive teams.
5. Behavioral complementarity selects only within the final band.
6. Once a stable agent lineage is committed, drift and peer collapse are controlled.

Diversity never compensates for failing competence constraints.

## 3. Data And Split Integrity

Training prompt updates use only the optimization training split. The fixed competence probe is sampled from that split and remains unchanged for the run. Validation selects the best epoch. The test split is evaluated only as configured, normally once after restoring the validation-selected prompts.

The supported task layer is centralized in `multi_dataset_diverse_rl/tasks.py`. The task-level runner reads `configs/task_level_comparison*.yaml` manifests and writes standardized CSV/JSONL results.

## 4. Solver And Aggregation

All active agents answer the same question. Solver traces should end with:

```text
FINAL_ANSWER: <answer>
```

The current V8 experiments use plurality aggregation. The system selects the answer with the largest vote count and applies the configured deterministic tie-break. Candidate counterfactuals and offline joint-team evaluation use the same canonical plurality implementation.

Training records include per-agent correctness, plurality correctness and margin, C1/C2 coverage depth, invalid traces, shared errors, and pivotal boundary events.

## 5. Target-Agent Selection

At each update window, `hybrid_competence_boundary_v2` combines:

- general target-agent errors;
- C1 and C2 creation opportunities;
- actual plurality pivotal fixes;
- shared-error and dominant-wrong residuals;
- competence state from the fixed probe.

Specialization strength controls the weight of residual and boundary repair. It does not enable or disable QD search. Even when specialization strength is zero, mechanism alternatives, niche retention, and joint team selection remain active.

## 6. Teacher-Critic-Student Evolution

Each selected beam parent enters the Teacher-Critic-Student pipeline:

1. Teacher creates a Socratic repair question from abstract diagnostics.
2. Critic audits it.
3. Rejected questions are rewritten from Critic feedback and audited again.
4. Student emits `task_specific_repair` and `mechanism_alternative` candidates.
5. JSON retry and syntax-only repair recover malformed Student output.

Teacher context includes the target lineage status, committed anchor mechanism, committed peer mechanisms, and rescue/shared-error residual support. An uncommitted agent is explicitly allowed to leave its old prompt mechanism. For a committed agent, alternatives preferentially explore structural variants near the anchor, but joint selection decides whether a true switch is justified.

The Student must provide executable mechanism steps. Persona changes, generic verification, wording changes, and renumbering do not count as mechanism alternatives.

## 7. Candidate Quality

Candidate evaluation replaces one target prompt while holding peer prompts fixed. The active competence and specialization reward can use:

- target-agent accuracy gain or loss;
- C1 and C2 net changes;
- actual plurality gain or loss;
- shared-error residual repair;
- boundary rescue;
- invalid and catastrophic-accuracy guards.

Global prompt embedding distance and unconditional novelty bonuses are not candidate reward terms. Before a lineage is committed, self-drift, cycle-to-self, and lineage-switch penalties are zero.

## 8. Search-Space Preservation

Quality constraints protect the active team, but they do not erase every novel branch immediately. Generated candidates first pass a cheap schema, completeness, length, duplicate, type, and mechanism-step prescreen. After candidate-batch evaluation they are separated into three states:

- `Safe`: valid, fully evaluated candidates without catastrophic target, C1, or C2 regression. Only Safe items may enter the long-term QD archive, joint representatives, lineage selection, or the active team.
- `Probation`: genuinely novel candidates with only bounded small regressions. They can be used as later TCS parents for at most two updates, but can never enter active-team selection directly.
- `Catastrophic`: invalid, duplicated, non-novel, or materially regressing candidates. These are discarded.

If the initial batch lacks two Safe non-incumbent candidates, a Safe task repair, or a Safe distinct mechanism, bounded refill runs for at most two rounds. The Teacher and Student receive structured rejection feedback, including accuracy/C1 losses and duplicate-niche causes, rather than a generic request to try again. Solver rollouts for already seen prompt-question pairs are reused.

The long-term Safe archive holds up to six niche elites per agent. `beam_size=3` means three representatives selected from that archive for joint enumeration, not an archive size limit. Parent A is the active prompt; Parent B is chosen round-robin from an underused Probation branch or Safe niche, ensuring archived niches receive reproduction opportunities.

Rescue, shared-error, and same-wrong measures are recomputed for every proposed team combination because they depend on the other four prompts. The fixed optimization probe is deterministically split into two folds. Joint selection uses `mean(fold diversity) - 0.5 * fold gap`, after incumbent integer loss floors and hierarchical vote, total-correct, bottom-2, C1, and C2 bands. This is narrower and more reproducible than retaining a large five-dimensional Pareto frontier.

At most three active prompts change in an early epoch and two in a later epoch; a single next-epoch relaxation is available only after repeated no-diversification selection. Lineages become provisional after one stable snapshot and committed after two. QD readiness can keep residual specialization at its 0.15 floor only when competence gates, two Safe niches, diversity, and fold stability all pass.

## 9. Mechanism Representation

`multi_dataset_diverse_rl/mechanisms.py` normalizes Student mechanism steps into an operation sequence. It removes persona names, shared solver prefixes, output-format text, step numbers, and generic careful/verify wording.

Known operations include:

```text
enumerate_candidates, extract_constraints, hard_elimination,
weighted_scoring, pairwise_comparison, counterfactual_check,
timeline_construction, binding_resolution, semantic_role_check,
syntactic_agreement_check, discourse_distance_check,
contradiction_minimization, evidence_accumulation,
option_elimination, final_consistency_check
```

Unknown but meaningful steps are preserved as normalized text.

Sequence distance is normalized Levenshtein distance over operation sequences. Embedding distance is `1 - cosine_similarity` over the normalized mechanism text, using the local embedding model. Full prompts are never embedded for mechanism comparison. Embeddings are cached by normalized mechanism hash.

```text
mechanism_distance = 0.50 * sequence_distance
                   + 0.50 * embedding_distance
```

Mechanism distance is secondary evidence. A near duplicate requires the same normalized sequence and embedding similarity of at least `0.97`.

## 10. Behavioral Profiles

Each unique beam prompt is evaluated on the fixed optimization probe. For each agent prompt, the system records:

```text
answer_vector
correctness_vector
error_vector
rescue_vector
unique_correct_vector
shared_error_vector
wrong_answer_cluster_vector
```

A rescue is a correct answer when at most one peer is also correct. A unique-correct case has no other correct peer. A shared error is an incorrect answer while at least two peers are also wrong.

Pairwise behavior distance is:

```text
0.50 * correct_set_jaccard_distance
+ 0.35 * shrinkage_adjusted_rescue_set_distance
+ 0.15 * shared_wrong_complementarity
```

Rescue-set distance is zero when both prompts have no rescue support. This prevents sparse evidence from appearing maximally diverse.

## 11. Per-Agent QD Archive

Each agent keeps a beam of three prompts. The beam is a quality-diversity archive, not fixed safe/exploit/explore roles.

The niche key is:

```text
(primary mechanism family, first four normalized operations)
```

Hard-guard failures cannot enter the archive. Within one niche, candidates are ranked by target accuracy, C1 net delta, C2 net delta, plurality gain, penalized reward, and earlier generation. Novelty cannot displace a higher-quality candidate in the same niche.

The archive retains:

1. the incumbent or strongest stable elite;
2. the best quality elite from a different niche;
3. a quality-passing elite maximally distinct from retained mechanisms and available behavior estimates.

Sources are logged as `incumbent`, `task_repair_niche`, and `mechanism_niche`. Any source may become active during joint selection.

## 12. Joint Active-Team Selection

At every epoch end, active prompts are selected jointly rather than greedily per agent.

With five agents and beam size three, the system evaluates at most 15 unique agent-prompt profiles on the fixed probe, then enumerates all `3^5 = 243` teams offline. No team-level solver calls are needed.

The probe cache key includes agent, prompt hash, solver model, question hash, parsing configuration, aggregation configuration, and seed.

For each team, the offline evaluator computes plurality accuracy, mean individual accuracy, bottom-2 accuracy, C1, C2, plurality margin, correctness vectors, and rescue profiles.

## 13. Quality Feasibility And Frontier

A team must remain within configured one-question tolerances of the incumbent on:

```text
plurality accuracy, mean accuracy, bottom-2, C1, C2
```

Each agent must also remain within `0.03` of its initial, incumbent, and committed-anchor accuracy where applicable.

The feasible teams are reduced to a five-dimensional epsilon-Pareto quality frontier. Diversity is evaluated only on that frontier. If no frontier survives, the incumbent team is retained.

## 14. Team Complementarity

For each frontier team:

```text
team_diversity_score =
    0.45 * mean_pairwise_behavior_distance
  + 0.25 * minimum_pairwise_behavior_distance
  + 0.20 * mean_pairwise_mechanism_distance
  + 0.10 * rescue_balance_score
```

The minimum pair term discourages a team where only some agents differentiate. Rescue balance is `1 - HHI` when rescue support exists and zero otherwise.

Before any committed anchors exist, the team with the highest diversity score on the quality frontier is selected without self-lineage penalties. This is the symmetry-breaking phase.

## 15. Stable Lineages

Each agent lineage is `uncommitted`, `provisional`, or `committed`.

- Two consecutive quality-passing epochs in the same or nearby mechanism form a provisional lineage.
- A third stable epoch commits it when behavior remains close and existing rescue support does not disappear.
- A commitment at the final training epoch is logged as committed but not subsequently exercised.

Only committed lineages receive drift constraints:

```text
lineage_drift = 0.50 * anchor_mechanism_drift
              + 0.50 * anchor_behavior_drift
```

Drift above `0.35` contributes a joint-selection penalty. Drift above `0.75` is rejected unless agent accuracy gains by at least `0.03` or team plurality gains by at least `0.02`.

Switching uses hysteresis. A different lineage is first recorded as pending and must be selected for two consecutive epochs before replacing the committed anchor. Returning to the anchor cancels the pending switch.

Committed peer anchors also impose a soft collapse penalty above similarity `0.85`. Same-sequence, behaviorally near-identical copies above similarity `0.97` cannot occupy another active position.

## 16. Stable Team Score

Within the quality frontier:

```text
stable_team_score = team_diversity_score
                  - mean_lineage_drift_penalty
                  - mean_peer_collapse_penalty
```

Tie-breaks are plurality, mean accuracy, bottom-2, C1, C2, fewer prompt changes, and prompt-hash order.

The diagnostic stable specialization score is:

```text
mean_behavior_distance
+ 0.5 * min_behavior_distance
+ 0.25 * mean_mechanism_distance
- 0.5 * mean_lineage_drift
```

It is not a candidate reward. Validation may use it only as the last tie-break after vote and competence fields.

## 17. Validation And Final Test

`vote_generalization_first` orders validation states by plurality vote, mean accuracy, bottom-2, C1, C2, plurality margin, invalid rate, stable-specialization tie-break, and earlier epoch. Final testing restores `best_prompts.json` selected from validation.

## 18. Checkpoint And Outputs

Checkpoint behavior fingerprints include all QD, mechanism, behavior, joint-quality, lineage, and peer-collapse settings. An old V8 checkpoint fails with:

```text
V8 behavior fingerprint mismatch: joint quality-diversity lineage policy changed
```

New diagnostic files are:

```text
quality_diversity_archive.jsonl
joint_team_selection_history.jsonl
behavior_profile_history.jsonl
lineage_history.jsonl
```

`run_meta.json` records all method versions and feature flags. `accuracy_results.csv/jsonl` exports stable specialization, lineage, joint-selection, niche occupancy, accuracy, plurality, coverage, and cost metrics.

## 19. Code Map

```text
multi_dataset_diverse_rl/system.py             orchestration and model calls
multi_dataset_diverse_rl/mechanisms.py         normalized mechanism representation
multi_dataset_diverse_rl/behavior_profiles.py  behavioral vectors and distances
multi_dataset_diverse_rl/quality_diversity.py  QD archive and offline team metrics
multi_dataset_diverse_rl/lineage.py             lineage state, drift, hysteresis
multi_dataset_diverse_rl/cli.py                 training, validation, checkpoints
scripts/experiment_config.py                    named experiment settings
scripts/run_task_level_accuracy.py              task-level runner
```

## 20. Boundaries

- The method optimizes prompts, not model weights.
- Fixed-probe behavior is an estimate and can overfit; validation remains separate.
- Mechanism extraction depends on Student-provided steps and normalization.
- Diversity is diagnostic and selection evidence, not proof of causal specialization.
- Multi-seed matched experiments are required for scientific conclusions.
