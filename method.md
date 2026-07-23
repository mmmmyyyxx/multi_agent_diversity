# Peer-State Counterfactual Prompt Optimization

`method_version = peer_state_counterfactual_v2`

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

For C0 examples, where `G = 0`, assignment still ranks oracle soft-utility gain and dominant-wrong departure before inertia and load balancing. A previous owner is retained only when its opportunity is within tolerance of the best opportunity. `owner_age_by_question` increases the non-C0 switch threshold and is written to diagnostics. A max-wait rule prevents any agent from being ignored indefinitely.

## 5. ProposalContext And TCS

Candidate generation uses only the target agent's active prompt as parent. The generator input is one of three distinct typed objects:

```text
AccuracyProposalContext       (B1)
PeerStateProposalContext      (B2 and B3)
ResponsibilityProposalContext (B4)
```

`AccuracyProposalContext` contains only individual errors, individual correct protection cases, and an accuracy/invalid-only update summary. `PeerStateProposalContext` contains `coverage_cases`, `conversion_cases`, peer/team state, preservation evidence, and a vote/competence summary, but has no assigned, owner, age, or responsibility fields. B3 may use responsibility internally to select evidence, but the generator cannot observe that fact. Only `ResponsibilityProposalContext` contains assigned residual cases, owner age, responsibility reason, responsibility summary, and assigned-repair history.

Teacher, Critic, and Student receive the same typed object. The Critic additionally sees the typed Teacher proposal; the Student additionally sees the approved proposal. Each context log records the concrete class, serialized top-level fields, recursive field paths, forbidden-field checks, and a hash of the shared serialized context. This provides runtime evidence that B1 cannot observe peer or responsibility state, B3 cannot observe ownership fields, and B4 can.

Every supplied case carries explicit derived state so the evaluator does not
have to infer the relationship between the current answer and gold:

```text
target_status
required_transition
team_vote_status       (peer-state contexts only)
case_role
repair_goal
forbidden_transition  (preservation cases)
```

The Teacher emits a testable repair hypothesis with six typed fields:

```text
observed_failure_pattern
generalizable_mechanism
decision_rule
uncertainty_or_abstention_rule
preservation_conditions
evidence_summary
```

Here, task-general means applicable to unseen examples within the current task.
It does not mean transfer across unrelated benchmarks, and it must not memorize
the supplied examples or gold answers.

The Critic is a legality and consistency gate, not a pre-rollout performance
predictor. Its hard checks cover context consistency, sample memorization,
executable behavior, internal consistency, preservation, output-contract
safety, explicit peer copying, forced resolution from occupational or social
stereotypes, and generic-only changes. Any failed hard check blocks Student.
Subjectivity, incomplete coverage, use of a common task method, overlap with the
parent prompt, and uncertain empirical benefit are recorded as soft concerns
and do not block rollout.

Before auditing, the Critic must copy every derived case fact into a typed
`case_fact_restatements` array. A missing or incorrect restatement invalidates
the Critic response and retries the same Teacher proposal. Approval is computed
only when every hard check passes and `blocking_reasons` is empty. The numeric
`score` is diagnostic and never controls approval.

Student output must contain exactly `num_candidates_per_parent` typed raw
candidates or the call enters JSON retry. A schema-valid candidate that copies
the text of a supplied optimization example is removed before Stage A.
Parent-identical and duplicate prompts may further reduce the post-schema pool.
The funnel separately records invalid Critic responses, requested/raw/schema
counts, sample-memorization rejection, non-parent prompts, and deduplication.

`tcs_rounds.jsonl` records every Teacher, Critic, and Student attempt separately:
request/response hashes, bounded response excerpts, JSON extraction, schema
validity, fact-restatement validity, hard checks, effective approval, diagnostic
score, blocking reasons, soft concerns, and Student count diagnostics.
Transport failures, evaluator state misreads, legitimate hard rejection, and
Student schema failure are therefore distinct.

Context limits default to six cases per category and 24,000 characters. Selection is deterministic and logs available, selected, truncated, character, and estimated-token counts.

## 6. Paired Candidate Rollout

Candidate evaluation replaces only the target agent's active prompt while holding all four peer profiles fixed:

```text
incumbent team = active five-prompt team
candidate team = incumbent team with target prompt replaced
```

A `PromptQuestionEvaluator` is shared by optimization, validation, and final
test. It uses an in-memory singleflight layer backed by a concurrent SQLite
cache at `<out_root>/_shared_solver_cache.sqlite`. The persistent key contains
solver model and endpoint identity, output-contract version, parser version,
temperature, maximum tokens, evaluation-replica seed, prompt-content hash, and
question-content hash. It contains neither agent id nor setting name. Thus
matched settings with the same seed use exactly the same observation for an
identical prompt-question pair; different seeds remain independent replicas.

SQLite rows use atomic `pending`/`ready` claims, WAL mode, dead-owner PID
detection, and stale-claim recovery. The cached value contains the raw response,
parsed answer, validity status, token usage, response/request identity, and
creation time.

Prompt canonicalization converts line endings and removes trailing line whitespace and outer blank lines. It preserves internal newlines, indentation, lists, and paragraphs. Prompt hashes and cache keys use this structure-preserving text.

The non-optimizable system layer appends a task-specific solver output contract.
For `option_letter`, the only permitted final payload is one uppercase option
letter with no punctuation or explanation. The contract version
`task_output_contract_v1` participates in solver request identity, cache keys,
RunIdentity, checkpoint compatibility, and run metadata. The optimized prompt
cannot remove or rewrite this contract.

Stage A evaluates every new candidate on a deterministic mixture of representative, coverage, conversion, and preservation examples. Specialized pools are selected first without overlap; representative sampling fills the remaining fixed total budget. Stage B completes only the shortlist on the full fixed probe. Funnel diagnostics record requested, available and selected pool sizes, removed overlap, and actual unique Stage A size.

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

Validation uses the same prompt-question evaluator as optimization and test, backed by the matched experiment's persistent cache. The same prompt-question pair is called once even when several agents or settings share a prompt or the pair appears in another evaluation phase.

Solver output is valid only when it has exactly one line-level `FINAL_ANSWER:` marker and the parsed answer belongs to the task domain. Metrics separately count `missing_final_answer`, `multiple_final_answers`, `unparseable_final_answer`, `out_of_domain_answer`, and `valid`. Every unique invalid prompt-question observation is written to `solver_invalid_outputs.jsonl` with raw final payload, marker count, bounded response excerpt, response hash, and request identity. The strict parser is not relaxed to hide instruction-following failures.

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

The run-specific preflight accepts manifest, tasks, settings, seeds, output root, and Config overrides. Before API use it verifies split existence, requested sizes, zero overlap, split hashes, role-specific API configuration, model names, candidate and Stage B budgets, TCS limits, call/token caps, output-directory identity, and successful RunIdentity construction.

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
tcs_rounds.jsonl
candidate_decisions.jsonl
solver_invalid_outputs.jsonl
llm_calls.jsonl
cost_summary.json
<out_root>/_shared_solver_cache.sqlite
```

`scripts/real_api_role_transport_smoke.py` independently tests solver, Teacher,
Critic, and Student schemas without forcing Student transport through the method
approval gate. `scripts/critic_calibration_replay.py` runs a human-labeled
evaluator-only set containing acceptable task-internal repairs and hard-blocker
examples. It requires at least one good proposal to pass, rejects all memorizing
proposals, validates every fact restatement, and makes zero Solver or optimizer
calls. `scripts/real_api_resume_smoke.py` performs a controlled
checkpoint interruption, resumes two copies of the same state through the
shared solver cache, and compares substantive final artifacts and cache hashes.
