# Peer-State Counterfactual Prompt Optimization

`method_version = peer_state_counterfactual_v1`

## 1. Problem Formulation

The system centrally optimizes a five-prompt ensemble for equal-weight plurality voting. Model weights are fixed. For each example, let:

```text
G = number of valid votes matching gold
H = size of the largest valid wrong-answer cluster
M = G - H
```

The canonical vote is correct exactly when `M > 0`. The optimization objective is vote improvement, per-agent competence is a hard constraint, and soft vote utility is only a dense search signal.

```text
Team Rollout
-> Peer-Conditioned Opportunity Attribution
-> Residual Responsibility Assignment
-> Responsibility-Conditioned TCS
-> Paired Candidate Rollout
-> Competence-Constrained Vote Update
```

The method does not optimize generic diversity, trace distance, prompt wording distance, or unconditional answer dispersion. Complementarity means that different agents repair different residual team errors.

## 2. Team And Peer State

`TeamVoteState` represents one complete five-agent rollout. It stores normalized answers, validity, correctness, the complete wrong-answer histogram, `G`, `H`, `M`, the plurality result, and tie diagnostics.

`PeerVoteContext` is a separate leave-one-out object for one target agent. It always contains exactly four peers. It is constructed from `TeamVoteState` by removing the target index before computing peer counts and histograms. Therefore a target vote cannot leak into a field named `peer_*`.

Invalid answers remain visible in the validity vector but do not vote. Under the canonical tie policy, a top tie returns an empty vote and counts as incorrect.

## 3. Oracle Repair Opportunity

`OracleRepairOpportunity` asks a hypothetical question before candidate generation:

```text
With the four peer answers fixed, what would happen if this target agent were repaired to gold?
```

It records direct vote-fix potential, oracle soft-utility gain, C0 coverage opportunity, dominant-wrong membership, and whether the agent currently supplies a unique or pivotal correct vote. This object is used only for responsibility assignment. It is not evidence that any generated prompt actually achieves the repair.

## 4. Residual Responsibility Assignment

Every currently vote-wrong example receives at most one primary owner, chosen only among agents that are currently wrong on that example. Ranking prioritizes:

1. direct vote repair;
2. oracle soft-utility gain;
3. departure from a dominant wrong cluster;
4. legal previous-owner inertia;
5. lower assigned load;
6. longer wait since selection;
7. deterministic hash;
8. agent id.

For C0 examples, where `G = 0`, assignment first preserves a legal previous owner and otherwise balances coverage load deterministically. `owner_age_by_question` increases the switch threshold, participates in C0 stability, and is written to diagnostics. A max-wait rule prevents any agent from being ignored indefinitely.

## 5. ProposalContext And TCS

Candidate generation uses only the target agent's active prompt as parent. `ProposalContext` contains that parent plus bounded sets of:

```text
assigned coverage cases
assigned conversion cases
preservation cases
representative cases
responsibility and previous-update summaries
```

Teacher, Critic, and Student receive the same typed context, including one explicit `context_policy`. The Critic additionally sees the typed Teacher proposal; the Student additionally sees the approved proposal. In B3, assigned cases are presented only as peer-state evidence. In B4, all three roles additionally interpret those same cases as owned residual responsibilities. Thus B3 does not silently inherit B4's responsibility-conditioned instructions.

The Critic applies policy-specific checks plus preservation, generic chain-of-thought, peer copying, answer memorization, parent-specific diagnosis, and procedural executability. Each TCS log records the context policy and a hash of the shared serialized context.

Teacher, Critic, and Student outputs are strict typed JSON. Critic approval requires a real JSON boolean and `score >= 0.75`. Missing fields, wrong types, parse failure, or a low score reject the proposal.

Context limits default to six cases per category and 24,000 characters. Selection is deterministic and logs available, selected, truncated, character, and estimated-token counts.

## 6. Paired Candidate Rollout

Candidate evaluation replaces only the target agent's active prompt while holding all four peer profiles fixed:

```text
incumbent team = active five-prompt team
candidate team = incumbent team with target prompt replaced
```

The fixed optimization probe caches prompt-question rollouts with singleflight. Stage A evaluates every new candidate on a deterministic mixture of representative, coverage, conversion, and preservation examples. Stage B completes only the shortlist on the full fixed probe.

For the round-robin peer-state ablation, coverage and conversion pools are global and do not disappear merely because no owner map exists. Representative examples are ordered by `seed + question_hash`, never by dataset order.

## 7. Candidate Marginal Contribution

`CandidateMarginalContribution` is computed from real candidate rollouts relative to the incumbent. It records vote gains and losses, net vote delta, soft-utility delta, coverage gains and losses, dominant-wrong exits and joins, and assigned-residual repairs.

`ProtectionContribution` separately records lost unique-correct and pivotal-correct cases. These realized objects are distinct from the pre-generation oracle opportunity. A C0 wrong-to-wrong label change receives zero soft gain.

## 8. Competence Constraints

`CandidateEvaluation` contains required typed fields:

```text
PromptCompetenceMetrics
TeamOutcomeMetrics
CandidateMarginalContribution
ProtectionContribution
```

The selector does not read missing fields as zero. A peer-state candidate must satisfy local and initial competence floors, invalid-output limits, vote-loss limits, unique-correct protection, and pivotal-correct protection. `ConstraintDecision` records every passed guard and every rejection reason.

## 9. Vote-First Selection

Stage A has accuracy, vote, and responsibility/coverage channels. Each channel first selects top-k. The merged set is ordered by cross-channel Pareto front, then the sum of channel ordinal ranks. Prompt hash is used only after all substantive indicators tie. The same rule fills unused budget, so there is no hash-based arbitrary filler.

Feasible Stage B candidates are ordered by:

```text
1. larger net vote delta
2. fewer vote losses
3. larger soft-utility delta
4. larger coverage gain
5. larger assigned-residual utility delta
6. larger target correct count
7. fewer invalid outputs
8. earlier generation
9. stable hash
```

A candidate is accepted for positive net vote gain, or for zero net vote change with no vote loss and sufficient soft gain, or for a clear competence gain with no unique/pivotal loss. C0 wrong-to-wrong changes are not accepted.

## 10. Online Responsibility Refresh

After an accepted prompt transaction, the system immediately recomputes team states, all four-peer contexts, oracle opportunities, owner assignments, and target-selection pressure. Both responsibility-enabled settings use online refresh. If refresh fails, the transaction restores `previous_active_prompt`; that transactional value is never a generation parent.

Update diagnostics include the complete candidate funnel, guard rejection counts, owner distribution, owner switches, owner ages, direct-fix/coverage/dominant-wrong assignment counts, and max-wait triggers.

## 11. Validation

`ValidationProbeEvaluator` has a separate cache keyed by validation probe hash, model identity, prompt hash, question hash, parser version, temperature, and seed. The same prompt-question pair is called once even when several agents share a prompt.

Validation first enforces per-agent and mean competence feasibility, then ranks states by plurality accuracy, net vote gain versus initialization, fewer vote losses, soft utility, lower C0, mean and minimum individual accuracy, invalid rate, and earlier epoch. The test split is not used for selection and is evaluated once after validation restores `best_prompts`.

## 12. Ablation Protocol

Method differences are explicit `ExperimentProtocol` fields rather than inferred boolean combinations.

| Setting | Target | Sample pool | TCS context | Candidate selection | Refresh |
|---|---|---|---|---|---|
| `shared_baseline` | none | none | none | none | off |
| `shared_independent_accuracy_tcs` | round-robin | individual errors | generic accuracy | individual accuracy | off |
| `shared_peer_state_credit_round_robin` | round-robin | global peer state | generic peer state | constrained vote-first | off |
| `shared_peer_state_responsibility` | dynamic responsibility | assigned residuals | generic peer state | constrained vote-first | online |
| `shared_peer_state_full` | dynamic responsibility | assigned residuals | responsibility-conditioned | constrained vote-first | online |

The B2-to-B3 change activates the dynamic responsibility subsystem, including assigned pools and refresh. B3-to-B4 changes only `tcs_context_policy`. All five settings share agent count, initialization, split, tie policy, candidate-generation count, Stage A/B budgets, validation, and test protocol.

## 13. Initialization And Tie Policy

The default initialization is `shared_identical`: all five agents start from the same prompt so that later specialization must arise from centrally assigned residual errors. `provided_prompt_set` is available only when exactly five non-empty prompts are supplied.

The canonical tie policy is `abstain`. Ties are incorrect and logged with `tie_count` and `tie_rate`. Other tie rules remain low-level analysis options but are rejected by the formal system.

The agents do not communicate directly. The method is centrally coordinated prompt ensemble optimization.

## 14. Reproducibility

`RunIdentity` records method and setting, git commit and dirty state, behavior-config fingerprint, manifest SHA, all split-file SHAs, and all split question-set hashes. The behavior fingerprint covers model and endpoint identity, temperatures, token limits, seed, initialization, prompts, tie and utility policies, responsibility parameters, TCS limits, candidate budgets, constraints, and parser/task identity.

The same identity is stored in `run_meta.json`, `training_checkpoint.json`, and the root experiment matrix. Resume and completed-run reuse compare every field and reject old or mismatched artifacts explicitly.

Core artifacts are:

```text
run_meta.json
training_checkpoint.json while interrupted
history.json
best_prompts.json
final_summary.json
peer_state_history.jsonl
responsibility_assignments.jsonl
tcs_context_history.jsonl
candidate_decisions.jsonl
llm_calls.jsonl
cost_summary.json
```
