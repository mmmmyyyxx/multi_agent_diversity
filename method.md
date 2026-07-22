# Peer-State Counterfactual Prompt Optimization

## 1. Problem Formulation

The system contains five fixed solver models with independently editable prompts. For question (x), agent (i) produces answer (a_i). Equal-weight deterministic plurality vote produces the team answer. Training changes prompts, not model weights or vote weights.

The optimization objective is team performance under an individual competence constraint:

```text
estimate each agent-example pair's counterfactual team value
assign unresolved team errors to primary Student owners
generate responsibility-conditioned prompt repairs
accept vote-first improvements only when competence is preserved
```

`method_version` is `peer_state_counterfactual_v1`. There is no runtime compatibility path for earlier methods.

## 2. Why Individual Prompt Accuracy Is Insufficient

An individually accurate prompt can repeat the same errors as its peers, while a prompt with the same individual accuracy can supply decisive correct minority votes. Candidate value therefore depends on the current four peer answers. Individual accuracy is enforced as a non-regression constraint; it is not the primary Stage-B objective.

## 3. Peer Vote State (G, H, M)

For each question:

```text
G = number of valid gold votes
H = largest valid wrong-answer cluster
M = G - H
```

The full `PeerVoteState` stores normalized answers, validity, correctness, the complete wrong-answer histogram, dominant wrong ties, the true plurality result, and tie diagnostics. Thus `A,A,B,B,B` and `A,A,B,C,D` remain distinct despite equal (G=2).

Search uses the dense utility:

```text
U(G, M) = 0                           when G = 0
U(G, M) = sigmoid(M / soft_vote_tau)  otherwise
```

Wrong-to-wrong changes at (G=0) have zero utility. Creating the first gold vote yields positive utility, lowering a dominant wrong cluster raises utility when a gold vote exists, and stable correct states saturate. Reported performance remains true plurality accuracy.

## 4. Counterfactual Agent-Example Credit

For target agent (i), peers remain fixed and only (a_i) is replaced by gold. The system records:

```text
direct_vote_fix
fix_soft_utility_gain
coverage_opportunity
dominant_wrong_member
unique_correct
pivotal_vote_correct
```

Unique and pivotal correctness use the same deterministic leave-one-out plurality policy as normal evaluation. No synthetic wrong answer is introduced.

## 5. Residual Responsibility Assignment

Each vote-wrong probe example receives one primary owner among currently wrong agents. Ranking is deterministic:

```text
direct vote fix
soft-utility repair value
dominant-wrong membership
previous-owner inertia
lower assigned load
longer wait since update
stable hash and agent id
```

An owner changes only for a newly available direct fix or a soft-utility advantage larger than `responsibility_switch_margin`. C0 assignments preserve legal owners and otherwise balance load deterministically. Target selection aggregates assigned direct fixes, utility, coverage, and dominant-wrong responsibilities. `responsibility_max_wait_updates` provides fairness. Round-robin exists only in named ablations.

## 6. Responsibility-Conditioned TCS

Teacher receives assigned coverage/conversion cases and unique/pivotal preservation cases. Each assigned case contains the question, gold, target answer, peer histogram, (G,H,M), dominant wrong answers, direct-fix value, soft gain, and responsibility reason.

Teacher proposes a generalizable executable decision-procedure repair. Critic rejects generic chain-of-thought, peer copying, memorization, preservation risk, preset roles, and surface wording changes. Student returns:

```json
{
  "candidate_prompt": "...",
  "target_failure_mechanism": "...",
  "repair_procedure": "...",
  "preservation_rule": "...",
  "expected_responsibility_effect": "..."
}
```

Candidate quality is determined by paired solver rollouts, not a candidate-text critic.

## 7. Multi-Channel Stage A

Every generated candidate runs on the same fixed Stage-A pool:

```text
12 representative
6 assigned coverage
6 assigned conversion
4 unique/pivotal preservation
```

Three independent channels retain top-k candidates:

```text
accuracy: target correct, lower invalid
vote: net vote delta, lower vote loss, soft utility
responsibility: coverage, assigned utility, lower unique/pivotal loss
```

Their union is deduplicated. Stage B has a fixed total budget shared by new and memory candidates. Candidates outside the shortlist never receive full-probe evaluation.

## 8. Competence-Constrained Vote-First Stage B

Stage B evaluates shortlisted candidates on the complete fixed optimization probe. Four peer profiles are reused and only the target prompt is run. A candidate must pass local and initial target-accuracy floors, invalid guard, vote-loss guard, unique-correct guard, and pivotal-correct guard.

Feasible candidates use this lexicographic order:

```text
net vote delta
lower vote loss
soft utility delta
coverage gain
assigned residual utility delta
target correct count
lower invalid count
earlier generation
stable prompt hash
```

Acceptance requires a strict improvement: positive net vote; or no vote loss plus sufficient soft gain; or no team loss plus an individual competence gain. Generic disagreement and C0 wrong-to-wrong changes cannot be accepted.

## 9. Behavioral Prompt Memory

Each agent has five semantic slots:

```text
active
competence_best
ensemble_best
responsibility_best
rollback
```

Active always remains. Rollback is not a normal generation parent. Behavior signatures include normalized answer hashes, correctness, invalidity, vote/coverage contributions, unique/pivotal correctness, dominant-wrong membership, ordered question hashes, and fixed-probe identity. After a peer update, every non-active memory prompt is re-evaluated against current peers before retention or parent selection.

## 10. Online Refresh

At most one agent changes per update. An accepted prompt activates immediately; the target fixed-probe profile, peer states, contribution signatures, memories, and residual responsibilities are then refreshed. Historical prompts from different agents are never combined.

## 11. Validation

Validation first rejects states that violate any initial per-agent accuracy floor, initial mean accuracy, or invalid-rate floor. Feasible states rank by:

```text
plurality vote accuracy
net vote gain vs initial
lower vote loss vs initial
mean soft vote utility
lower C0
mean individual accuracy
minimum individual accuracy
lower invalid rate
earlier epoch
```

Final test restores only validation-selected prompts. Test examples never participate in generation, candidate ranking, rollback, or best-state selection.

## 12. Ablations

```text
shared_baseline
shared_independent_accuracy_tcs
shared_peer_state_credit_round_robin
shared_peer_state_responsibility
shared_peer_state_full
```

The complete method adds responsibility-conditioned TCS and online responsibility refresh. Old setting names fail explicitly.

## 13. Outputs

Core audit streams are `peer_state_history.jsonl`, `responsibility_assignments.jsonl`, `candidate_decisions.jsonl`, and `prompt_memory_history.jsonl`. `run_meta.json` states that true plurality and peer wrong histograms are used, while generic diversity, trace diversity, team enumeration, and legacy compatibility are disabled.

## 14. Reproducibility

The fixed probe has a versioned content hash. Prompt-question solver calls are cached and singleflight-coalesced. Checkpoint version 1 stores prompts, memories, profiles, cache, responsibility owners/ages/loads, update cursor, histories, and random state. Any method, checkpoint, probe-version, or probe-hash mismatch fails and requires a new run.

Before experiments:

```powershell
$PY = "D:\Anaconda\envs\DL\python.exe"
& $PY -m pytest -q
& $PY -m compileall multi_dataset_diverse_rl scripts tests
git diff --check
& $PY scripts/preflight_peer_state.py --workspace .
```
