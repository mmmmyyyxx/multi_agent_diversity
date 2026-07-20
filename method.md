# Method: Accuracy-First Rollout-Diversity Prompt Optimization

## 1. Purpose

This repository evolves prompts for a fixed team of solver agents. It does not train model weights. Candidate reward ranks prompts during search, validation selects the best epoch, and final test runs after restoring that validation-selected state.

The current V8 rollout-only settings are:

```text
shared_accuracy_rollout_embedding_tcs
method_version = v8_accuracy_rollout_embedding

shared_vote_ready_rollout_diversity_tcs
method_version = v8_rollout_qd_vote_ready
```

The first setting is an accuracy-plus-rollout-diversity ablation. The second is the main vote-ready method. Both use the same models, strict splits, candidate budget, fixed optimization pool, archive capacity, validation protocol, and final-test protocol.

## 2. Core Principle

The new methods optimize:

```text
target-agent accuracy
+ Vote and C2-to-C3 conversion
+ useful behavior observed in real solver rollouts
```

They do not use prompt wording, prompt embeddings, optimizer-reported mechanism names, or artificial capability families as diversity evidence. A prompt is behaviorally different only when its fixed-probe solver outputs differ.

Quality is ordered before diversity:

1. Accuracy, invalid, Vote-loss, and C3-loss guards reject unsafe candidates.
2. Candidate accuracy and vote-readiness determine quality.
3. Rollout diversity breaks ties or differentiates quality-compatible candidates.
4. Joint selection prioritizes Vote, C3, total correctness, bottom-2, and margin before diversity.

Random wrong answers, empty traces, invalid outputs, and low-accuracy prompts cannot earn positive diversity.

## 3. Data Protocol

Prompt optimization uses only the training split. A deterministic fixed probe is selected from the configured candidate-evaluation pool and remains unchanged for the run. Validation is separate and selects the best epoch. Test is not used for optimization or selection and normally runs once after best-state restoration.

Strict manifests record split hashes and reject normalized-question overlap. The task layer is centralized in `multi_dataset_diverse_rl/tasks.py`.

## 4. Solver And Aggregation

All active agents answer the same question. Solver traces should end with:

```text
FINAL_ANSWER: <answer>
```

Current experiments use plurality: the answer with the largest vote count wins, with the configured deterministic tie-break. Candidate counterfactuals, offline team enumeration, validation, and test use the same canonical implementation.

Each rollout records answers, traces, correctness, invalid flags, Vote correctness, gold vote count, largest wrong cluster, and gold plurality margin.

## 5. Update Targets

At each `update_every` window, the target selector uses observed target errors, invalid output, pivotal vote opportunities, and dominant wrong-answer participation. New rollout-QD settings do not use capability profiles, capability HHI, mechanism families, or lineage status to select agents.

## 6. Candidate Generation

Each selected parent receives two generation channels under the configured budget:

1. `teacher_critic_student`: Teacher creates a Socratic accuracy-repair question from abstract rollout diagnostics; Critic audits it; Teacher can revise it using Critic feedback; Student writes candidate prompts.
2. `open_rollout_exploration`: Student directly proposes a prompt that may improve target accuracy, C2-to-C3 conversion, Vote recovery, margin, or dominant-wrong reduction.

Neither channel asks for mechanisms, capability labels, personas, or prompt-text novelty. They cannot see gold answers, concrete sample text, answer labels, or full peer prompts.

The minimal candidate schema is:

```json
{
  "candidate_prompt": "...",
  "target_error_pattern": "...",
  "accuracy_repair_rule": "...",
  "expected_accuracy_effect": "...",
  "rollout_diversity_intent": "..."
}
```

`rollout_diversity_intent` is generation context only. It is never used as a score or behavior label. JSON retry and syntax-only repair remain enabled for malformed Student output.

## 7. Candidate Evaluation

For target agent `i`, evaluation compares on the same candidate batch:

```text
baseline team  = current active prompts
candidate team = current active prompts with agent i replaced
```

Recorded transitions include:

```text
C0->C1, C1->C2, C2->C3
C3->C2, C2->C1, C1->C0
Vote gain and Vote loss
gold-margin gain and loss
dominant-wrong break and creation
```

`C2->C3` is the highest-value depth transition. `C3->C2` and Vote-correct to Vote-wrong are protected regressions.

Candidate evaluation reuses recorded prompt-question rollouts when available. Rollout metrics are computed from those results and add no evaluator-model calls.

## 8. Fixed-Probe Rollout Profile

Every retained prompt is evaluated on the fixed optimization probe and stores:

```text
answer_vector
correctness_vector
invalid_vector
trace_embedding_vector_per_question
rollout_signature_hash
```

The rollout signature hashes answer, correctness, and invalid vectors. Two textually different prompts with the same rollout signature are behavior duplicates. Textually similar prompts with different signatures may coexist.

Trace embeddings use the configured local sentence-transformer, currently `BAAI/bge-small-en-v1.5`. Only valid traces participate in positive trace distance, and a trace pair is counted only when at least one agent is correct.

## 9. Rollout Distance

Pairwise rollout distance is:

```text
D_rollout = 0.50 * D_correct_set
          + 0.20 * D_useful_wrong
          + 0.30 * D_valid_trace
```

`D_correct_set` is Jaccard distance between the sets of correctly answered probe questions. If both sets are empty, distance is zero.

`D_useful_wrong` measures different valid wrong answers only when the team already has two gold votes or the candidate improves Vote, gold margin, or dominant-wrong concentration. All-wrong random dispersion is not useful diversity.

`D_valid_trace` is mean cosine distance between valid solver-trace embeddings on supported questions. Empty, truncated, repetitive, or otherwise invalid traces contribute zero.

## 10. Quality Guards

A candidate is Safe only when:

```text
candidate_target_accuracy >= baseline_target_accuracy - 0.02
candidate_invalid_rate    <= baseline_invalid_rate + 0.02
C3->C2 loss count         <= 0
Vote loss count           <= 0
```

The thresholds are explicit configuration and checkpoint fields. Guard relaxation is not silent. The initial rollout-QD settings use zero Vote and C3 losses.

## 11. Candidate Objectives

The accuracy-rollout ablation ranks Safe candidates with:

```text
R_simple = 1.00 * target_accuracy
         + 0.20 * rollout_diversity_delta
         - 1.00 * positive_invalid_delta
```

The vote-ready method uses:

```text
R_vote_ready = 1.00 * target_accuracy
             + 1.00 * net_vote_rate
             + 1.00 * net_C3_rate
             + 0.30 * gold_margin_delta
             + 0.30 * dominant_wrong_net_rate
             + 0.15 * rollout_diversity_delta
             - 1.00 * positive_invalid_delta
```

Vote-ready candidate ordering is lexicographic: fewer Vote losses, fewer C3 losses, more Vote gains, more C2-to-C3 gains, higher target accuracy, larger margin gain, more wrong-cluster breaks, more rollout diversity, lower invalid rate, then earlier generation.

## 12. Rollout Archive

Each agent maintains a Safe archive of up to six candidates. The archive first deduplicates by prompt hash, then by rollout signature. Mechanism niche, mechanism novelty, prompt distance, and capability labels do not participate.

Archive retention covers the incumbent, highest-quality candidates, vote-ready and C2-to-C3 utility, and marginal rollout distance under the quality guard. At most three representatives enter joint enumeration:

1. current active prompt;
2. highest-quality Safe candidate;
3. highest marginal rollout-distance candidate that remains Safe.

The parent pool is active plus rollout representatives. It does not call the historical mechanism-niche parent selector.

## 13. Joint Team Selection

Each unique agent-prompt representative is solver-evaluated once on the fixed probe. With three representatives and five agents, at most 15 prompt profiles support offline enumeration of `3^5 = 243` teams. Team combinations require zero solver calls.

The vote-ready joint key is:

1. Vote-correct count;
2. C3-correct count;
3. total individual-correct count;
4. bottom-2 correct count;
5. mean gold plurality margin;
6. lower dominant-wrong concentration;
7. rollout diversity;
8. C2;
9. C1.

Thus a lower-Vote or lower-C3 team cannot win merely through high diversity. The accuracy-rollout ablation instead leads with total individual correctness, then Vote and C3, while retaining the same rollout-only evidence.

After joint selection changes an active prompt, the system synchronizes prompt history, rollout-signature history, accepted rollout archive, active candidate source, candidate funnel, and selected fixed-probe profile. It does not update capability profiles or mechanism lineage.

## 14. Validation And Final Test

`rollout_vote_first` selects the best epoch by:

```text
Vote, C3, mean individual, bottom-2, Oracle-to-Vote conversion,
gold margin, lower wrong concentration, lower invalid rate,
rollout diversity, earlier epoch
```

Final test restores `best_prompts.json` and evaluates that state once. `best_prompts.json` is the authoritative final prompt set.

## 15. Diagnostics And Integrity

Split summaries include C0 through C5 counts, C1/C2 vote success and failure, C3, Oracle-to-Vote conversion, margins, wrong concentration, same-wrong rate, invalid rate, and trace diversity.

New-method metadata explicitly reports:

```json
{
  "mechanism_diversity_enabled": false,
  "mechanism_metadata_required": false,
  "mechanism_distance_used_for_selection": false,
  "mechanism_based_decision_count": 0,
  "capability_labeling_enabled": false,
  "capability_profile_per_agent": null,
  "top_capability_family_per_agent": null,
  "prompt_text_diversity_used": false
}
```

The candidate funnel separates `teacher_critic_student`, `open_rollout_exploration`, incumbent, and historical channels.

## 16. Checkpoint And Resume

Checkpoint v6 stores prompts, rollout profiles, rollout-signature history, archive state, active sources, candidate funnel, caches, counters, random state, and all rollout objective/guard weights. Behavior-affecting mismatches fail before continuing; they never silently restart in the same output directory.

Completed runs remove `training_checkpoint.json`. Interrupted runs resume only with the same setting, split, seed, model, sizes, candidate budget, and behavior-affecting arguments.

## 17. Historical Compatibility

The completed historical setting remains available without semantic changes:

```text
shared_vote_tcs_competence_depth2_progressive_residual_hybrid
method_version = v8_stable_qd_lineage
```

Its mechanism schema, mechanism archive, capability diagnostics, lineage, and checkpoint parsing remain in the repository for reproduction and old-run analysis. New rollout-QD method versions do not enter those decision paths.

## 18. Code Map

```text
multi_dataset_diverse_rl/rollout_diversity.py       rollout distance, guards, archive, team keys
multi_dataset_diverse_rl/optimization/              TCS/Open generation and update pipeline
multi_dataset_diverse_rl/evaluation/                candidate and dataset evaluation
multi_dataset_diverse_rl/qd/joint_controller.py     fixed-probe offline team selection
multi_dataset_diverse_rl/persistence/               run metadata and checkpoint v6
multi_dataset_diverse_rl/cli.py                     train/validation/final-test lifecycle
scripts/experiment_config.py                        named settings
scripts/run_task_level_accuracy.py                  matched task runner
```

## 19. Boundaries

- Prompt optimization is not model-weight training or policy-gradient RL.
- Rollout diversity is empirical behavior evidence, not proof of causal specialization.
- Fixed-probe selection can overfit; validation and multiple matched seeds remain necessary.
- Oracle coverage alone is not the objective; the main question is whether correct information converts into plurality Vote.
