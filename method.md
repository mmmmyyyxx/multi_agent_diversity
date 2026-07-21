# Method: State-Conditioned Correlated-Error Optimization

## 1. Purpose

This repository evolves prompts for a fixed team of solver agents. It does not update model weights. Training reward and selection rank candidate prompts, validation selects the best state, and final test runs only after restoring that validation-selected state.

The current method is:

```text
method_version = v9_state_conditioned_error
reward_mode = rollout_state_conditioned
candidate_selection_mode = state_conditioned_accuracy_first
active_team_selector_version = state_conditioned_joint_v1
best_state_selection_mode = state_conditioned_vote_first
```

The main setting is `shared_state_conditioned_error_tcs`. The matched ablations are:

```text
shared_v9_accuracy_only
shared_v9_accuracy_coverage
shared_v9_accuracy_coverage_c2split
shared_v9_accuracy_coverage_c2split_trace_tiebreak
```

V8 method versions and completed runs retain their original semantics. V9 has separate configuration, selectors, metadata, and checkpoint compatibility checks.

## 2. End-To-End Data Flow

```text
training rollout window
  -> classify each question by current correct-agent count
  -> select target agents and build three repair routes
  -> Teacher-Critic-Student and open rollout generation
  -> candidate counterfactual evaluation on three disjoint pools
  -> hard quality guards
  -> accuracy epsilon band
  -> per-agent four-slot Archive
  -> fixed-probe prompt profiles
  -> offline joint team enumeration
  -> validation accuracy guard and vote-first selection
  -> restore best prompts
  -> final test
```

Only the optimization training split supplies search evidence. Validation selects states. Test is never used for prompt generation, candidate ranking, Archive retention, or joint team selection.

## 3. Solver And Plurality Vote

All five agents answer the same question. Solver traces should end with:

```text
FINAL_ANSWER: <answer>
```

The canonical aggregator is plurality: the normalized answer with the largest count wins. Top ties use the configured deterministic tie-break. Training diagnostics, candidate counterfactuals, joint enumeration, validation, and test all use this implementation.

For every question, V9 records:

```text
G = number of agents answering correctly
H = largest wrong-answer vote count
M = G - H
state = C0, C1, C2, or C3PLUS
```

Definitions:

```text
C0: G == 0
C1: G == 1
C2: G == 2
C3PLUS: G >= 3
```

The task abstraction also reports `option_count` for multiple-choice questions. This supports theoretical C2 rescuability without task-specific logic in the candidate evaluator.

## 4. Why The Objective Is State-Conditioned

Correct-set distance, wrong-answer dispersion, and trace distance have different meanings. V9 therefore does not combine them into a global rollout-diversity reward.

### C0 And C1

C0 and C1 optimize new correct coverage:

```text
C0 -> C1: an all-agent failure gains at least one correct solver
C1 -> C2: a singly covered question gains a second correct solver
```

Reverse transitions are losses. Merely changing one wrong answer into another earns nothing.

### C2

C2 has two separate repair paths:

1. If the target changes from wrong to correct, `C2 -> C3` records a third correct vote. It is not also counted as wrong-answer diversity.
2. If the target remains wrong and `G` remains 2, V9 may count a reduction in the largest wrong cluster:

```text
c2_wrong_cluster_reduction = max(0, baseline_H - candidate_H)
c2_wrong_cluster_creation  = max(0, candidate_H - baseline_H)
```

Wrong-answer dispersion is a task utility only in this second C2 case. C0, C1, and C3PLUS always receive zero wrong-dispersion task gain.

For `K` answer options, three wrong agents have theoretical minimum largest wrong cluster:

```text
H_min = ceil(3 / (K - 1))
```

This classifies C2 cases as strictly rescuable, tie-only rescuable, or unrescuable by dispersion. For example, four options can form `2:1:1:1`; three options can generally reach only `2:2:1`.

### C3PLUS

C3PLUS is already on the correct side of the five-agent plurality boundary. Different wrong answers or different traces do not create task reward. Correctness, invalid output, and regression guards remain active.

## 5. Candidate Generation Routes

V9 keeps the existing generation architectures:

```text
teacher_critic_student
open_rollout_exploration
```

Each generated candidate also carries one state-conditioned route:

```text
general_accuracy
coverage_repair
vote_conversion
```

`general_accuracy` repairs target-agent errors while preserving correct behavior. `coverage_repair` targets deterministically assigned C0/C1 residual cases. `vote_conversion` first seeks C2-to-C3; only when the target remains wrong may it reduce duplicated wrong votes on a rescuable C2 case.

C0/C1 cases are assigned to one or two agents by a deterministic hash of the question, seed, and agent ID. This creates different repair histories without fixed personas, capability names, or predefined roles. Assignments and rescue counters are checkpointed.

Teacher proposes a Socratic repair question. Critic audits it. If rejected, Teacher receives Critic feedback and rewrites it. Student then emits strict JSON candidates. JSON retry and syntax-only repair remain enabled. Prompts, not optimizer self-reported claims, are evaluated.

## 6. Three-Pool Candidate Evaluation

Each candidate replaces one target agent while all peer prompts remain active:

```text
baseline team  = current active prompts
candidate team = current active prompts with target agent replaced
```

The evaluation batch contains three disjoint pools:

```text
representative pool: natural optimization-split sample
coverage pool: known current C0/C1 cases
conversion pool: known current C2 cases, stratified by option count
```

If a targeted pool is short, representative examples fill the remaining budget. Actual pool counts, state counts, and option-count histograms are logged.

Only the representative pool estimates target accuracy and invalid rate for quality guards and accuracy ordering. Targeted-pool accuracy is diagnostic and cannot replace natural-distribution accuracy. Coverage and conversion pools estimate their corresponding state transitions.

Every per-question counterfactual row stores baseline and candidate state, G, H, M, option count, target correctness, vote correctness, tie status, target answer, and dominant wrong answer.

## 7. Hard Quality Guards

A non-incumbent candidate is Safe only when:

```text
candidate_target_accuracy >= baseline_target_accuracy - accuracy_guard_epsilon
candidate_invalid_rate    <= baseline_invalid_rate + invalid_guard_epsilon
C1 -> C0 loss count       <= state_c1_to_c0_loss_epsilon
C2 -> C1 loss count       <= state_c2_to_c1_loss_epsilon
C3 -> C2 loss count       <= state_c3_to_c2_loss_epsilon
Vote loss count           <= state_vote_loss_epsilon
```

The incumbent is always retained as a fallback. Failed candidates are catastrophic and cannot enter the V9 Archive.

V9's scalar `reward` field is only representative-pool target accuracy for compatibility and logging. It is not used to add state utilities together.

## 8. Accuracy-First Candidate Selection

Among Safe non-incumbents:

```text
best_accuracy = max(candidate_target_accuracy)
accuracy_band = candidates with accuracy >= best_accuracy - state_accuracy_tie_epsilon
```

Coverage and conversion slots can select only from this band. A weaker candidate cannot enter the Archive because it has large state utility or trace distance.

The quality keys are:

```text
global accuracy:
  target accuracy, lower invalid, wrong->correct, fewer correct->wrong,
  optional final trace tie-break, earlier generation, stable hash

coverage:
  C0->C1, C1->C2, fewer C1->C0, fewer C2->C1,
  target accuracy, lower invalid, optional final trace tie-break, stable hash

conversion:
  C2->C3, strict split gains, vote gains, wrong-cluster reduction,
  tie gains, fewer dominant-wrong creations, target accuracy, lower invalid,
  optional final trace tie-break, stable hash
```

Trace distance never appears before a quality metric.

## 9. Per-Agent Archive

Each agent keeps up to four rollout representatives:

```text
1. incumbent
2. overall accuracy best
3. coverage repair best, when enabled
4. vote conversion best, when enabled
```

Duplicate prompt hashes and duplicate rollout signatures are stored once. If one prompt wins multiple slots, the next eligible accuracy-band prompt fills capacity. No mechanism niche, prompt-text distance, persona, or capability label participates.

With four representatives per agent, five-agent offline enumeration has at most `4^5 = 1024` combinations. Prompt profiles are solver-evaluated once on the fixed optimization probe; team combinations add no solver or evaluator-model calls.

## 10. Joint Team Selection

First, teams enter a total-correct quality band:

```text
best_total_correct = max(team total_agent_correct_count)
allowed_slack = round(probe_size * num_agents * state_joint_total_correct_slack_rate)
keep teams with total_correct >= best_total_correct - allowed_slack
```

Within the band, the full V9 key is:

```text
lower C0,
more vote-correct,
more coverage depth C2,
more C2 strict vote-correct,
more C2 vote-correct,
lower C2 largest-wrong count,
higher bottom-2 correctness,
higher mean gold plurality margin,
lower invalid count,
optional trace distance as the final semantic tie-break,
stable prompt-hash tie-break
```

The ablation switches remove their terms from both Archive and joint selection:

```text
accuracy_only: no coverage, C2 split, or trace key
accuracy_coverage: coverage key only
accuracy_coverage_c2split: coverage and C2 key
accuracy_coverage_c2split_trace_tiebreak: full key, trace last
```

## 11. Validation And Final Test

Before training, V9 evaluates validation once and stores `initial_validation`. A later epoch is eligible only when:

```text
mean_individual_acc >= initial_validation_mean_individual_acc
                       - state_validation_accuracy_guard_epsilon
```

Eligible states are ordered by:

```text
Vote accuracy,
mean individual accuracy,
lower C0 rate,
C2 vote-correct rate,
bottom-2 accuracy,
mean plurality margin,
lower invalid rate,
earlier epoch
```

The initial state remains a valid fallback. Final test restores `best_prompts.json`, which is the authoritative final prompt set.

## 12. Metrics And Artifacts

Task objective metrics include Vote, mean/best/bottom-2 individual accuracy, Oracle, C0/C1/C2/C3PLUS, C2 vote correct/fail, C2 strict/tie, plurality margin, invalid rate, persistent/new/resolved C0, and all-agents-same-wrong rate.

Search diagnostics include directional transition counts, C2 wrong-cluster reduction/creation, theoretical C2 rescuability, fixed-probe trace distance, candidate pool composition, optimization route, coverage assignments, per-agent rescue counters, Archive slots, and joint quality-band size.

`state_search_diagnostics` accumulates directional transitions over evaluated candidates. It is reported separately from final validation/test task metrics and must not be interpreted as a held-out accuracy estimate.

`correct_set_rollout_distance_diagnostic`, `c2_wrong_answer_dispersion`, and `trace_embedding_distance` remain separate diagnostics. V9 never uses the historical composite rollout distance for reward or selection.

Important artifacts:

```text
run_meta.json
history.json
best_prompts.json
prompt_history.json
update_logs.jsonl
solver_rollout_records.jsonl
joint_team_selection_history.jsonl
training_checkpoint.json while incomplete
final_summary.json
cost_summary.json
```

## 13. Checkpoint And Resume

The global checkpoint format remains v6 for V8 compatibility. V9 additionally requires:

```text
state_conditioned_checkpoint_version = 1
```

V9 behavior fields, pool sizes, switches, policy versions, assignments, counters, random state, prompt profiles, and Archive state are fingerprinted or persisted. Missing or incompatible state causes an explicit resume error; the runner does not silently restart in the same output directory.

The historical default behavior fingerprint remains:

```text
48c2f27cdcda64d2f7b32d008957b4903c683f49012988c4e5cab301ed29d5fa
```

## 14. Historical V8 Compatibility

These settings remain available unchanged:

```text
shared_accuracy_rollout_embedding_tcs
method_version = v8_accuracy_rollout_embedding

shared_vote_ready_rollout_diversity_tcs
method_version = v8_rollout_qd_vote_ready

shared_vote_tcs_competence_depth2_progressive_residual_hybrid
method_version = v8_stable_qd_lineage
```

V8 may use its recorded composite rollout distance, mechanism schema, lineage, or historical selectors according to its method version. V9 branches before those decisions. Old output directories and compatible checkpoints remain readable.

## 15. Code Map

```text
multi_dataset_diverse_rl/state_conditioned.py          V9 states, transitions, Archive, team keys
multi_dataset_diverse_rl/optimization/target_selector.py state-routed case construction
multi_dataset_diverse_rl/optimization/candidate_generator.py TCS/open generation
multi_dataset_diverse_rl/evaluation/candidate_evaluator.py counterfactual and pool metrics
multi_dataset_diverse_rl/qd/joint_controller.py        fixed-probe team enumeration
multi_dataset_diverse_rl/evaluation/dataset_evaluator.py split summaries
multi_dataset_diverse_rl/persistence/checkpoint.py     fingerprint and resume checks
multi_dataset_diverse_rl/config.py                     config and CLI schema
scripts/experiment_config.py                           named matched settings
scripts/run_task_level_accuracy.py                     task-level runner
tests/test_state_conditioned_error_v9.py               V9 invariants
```

## 16. Verification Order

```powershell
$PY = "D:\Anaconda\envs\DL\python.exe"
& $PY -m pytest tests/test_rollout_qd_vote_ready.py -q
& $PY -m pytest tests/test_state_conditioned_error_v9.py -q
& $PY -m pytest -q
& $PY -m compileall multi_dataset_diverse_rl scripts tests
git diff --check
```

After tests, run an offline replay, then a single-task seed42 smoke, then the four matched seed42 settings. Multi-task and seed43/44 experiments should wait until those audits pass.
