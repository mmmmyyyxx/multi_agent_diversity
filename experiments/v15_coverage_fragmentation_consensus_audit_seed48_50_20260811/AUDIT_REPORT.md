# v15 Coverage Fragmentation → Consensus Conversion Offline Trajectory Audit

## 1. Executive Summary

The audit passes. Exact accepted-state replay recovered 66 train states and 57 accepted transitions across seeds 48–50 and S0/S1/S2. Every replayed state matched the recorded 75-example G histogram, oracle count, and repository plurality-vote count. Final vote/member counts and accepted-update counts matched the published train-only table.

The central result has two parts:

1. **Module1 has a strong descriptive specialization signature.** Relative to S0, S1 produced more previously uncovered discoveries, higher final oracle coverage, lower member-correctness correlation, and higher effective ensemble size in all three seeds.
2. **That specialization was disproportionately fragmented.** S1 produced more persistent singleton coverage in all three seeds, expanded the final oracle-vote gap in all three seeds, and improved orphan vote-conversion rate over S0 in only one of three seeds.

Module2 is useful but not stable. S2 improved orphan reinforcement, orphan-to-vote conversion, conversion latency, persistent-singleton rate, and vote yield relative to S1 in seeds 48 and 49. Seed50 reversed the rate/yield directions even though its final train vote rose. The correct verdict is therefore **mixed / partially supported**, not strong.

The data do **not** support the narrow claim that unique ownership simply removes discovered residuals from all other members and prevents reinforcement. Before episode resolution, a different member later received the residual in 48/51 S1 orphan episodes and 42/48 S2 episodes. The more specific gap is that only 24/48 S2 orphan episodes appeared in a later different-member responsibility-conditioned context, and late valid candidates often failed common-safe constraints.

## 2. Audit Scope and Provenance

- Git HEAD at replay: `48c2a133cfb294b94a10f1931cc061298dca2e17`
- Seeds: 48, 49, 50
- Settings: S0 Generic Evolution, S1 Member-Aware Dual-Target, S2 Responsibility-Conditioned Dual-Target
- Time axis: initial state plus accepted team states only
- API/model/training/test calls: 0
- Test artifacts consumed: false
- Method source modifications: 0
- Images generated: 0, by explicit user request

The published report tree does not contain the per-example train trajectories needed for this audit. The replay therefore read the existing raw **train-only** `peer_state_history.jsonl`, decisions, responsibility, context, selector, and dynamics files under `runs/v15f48`, `runs/v15f49`, and `runs/v15f50`. It did not read caches, checkpoints, final-test roots, test predictions, or test splits. Every consumed artifact and SHA256 is listed in `audit_manifest.json`.

## 3. Exact Definitions

- `G`: count of individually correct agents among five.
- Oracle correct: `G > 0`.
- Team vote: repository `build_team_vote_state`, plurality voting, top tie abstains and is incorrect.
- Repair distance `r`: the smallest subset of currently wrong agents which, when counterfactually repaired to gold, makes the exact repository aggregator vote correctly. All wrong-agent subsets were enumerated.
- Discovery: `G: 0 → >0`; singleton discovery is `0 → 1`.
- Orphan creation: `0 → 1` and child vote remains wrong.
- Reinforcement: a created orphan later reaches `G >= 2` before its coverage is lost.
- Consensus conversion: vote wrong → vote correct.
- An orphan episode ends on first vote conversion, first return to `G=0`, or run end. A later rediscovery is a separate episode.

Because an accepted transition changes only one target member, `|ΔG| <= 1`. The replay asserts that all non-target member answer/validity fields remain unchanged and that impossible `0→2+` and `1→3+` transitions are zero.

## 4. Replay Validation

| Check | Result |
|---|---:|
| Accepted states replayed | 66 |
| Accepted transitions replayed | 57 |
| State replay mismatches | 0 |
| G-histogram sum | 75 in every state |
| Oracle-count mismatches | 0 |
| Repository-vote mismatches | 0 |
| Published final train vote/member mismatches | 0 |
| Published accepted-update mismatches | 0 |

The exact accepted updates were S0/S1/S2 = 4/5/7 for seed48, 5/8/9 for seed49, and 3/7/9 for seed50.

## 5. Cross-Seed Mechanism Comparison

### 5.1 Module1: S0 → S1

| Seed | Discoveries S0→S1 | Final oracle S0→S1 | Final vote S0→S1 | Gap S0→S1 | Persistent singleton rate S0→S1 | Orphan conversion rate S0→S1 |
|---:|---:|---:|---:|---:|---:|---:|
| 48 | 14→17 | 64→67 | 57→59 | 7→8 | .286→.412 | .643→.588 |
| 49 | 17→22 | 61→71 | 56→56 | 5→15 | .000→.455 | .588→.500 |
| 50 | 10→12 | 61→63 | 55→56 | 6→7 | .000→.167 | .800→.833 |

Direction consistency is unusually clear for specialization: discoveries, final oracle coverage, and oracle-vote gap all increased in 3/3 seeds; pairwise correctness correlation fell and effective ensemble size rose in 3/3 seeds. Vote conversion did not keep pace: the orphan conversion rate fell in 2/3 seeds, and persistent singleton rate rose in 3/3.

The three-seed descriptive means moved from S0 to S1 as follows: discoveries 13.67→17.00; final oracle 62→67; final vote 56→57; oracle-vote gap 6→10; pairwise correlation .641→.550; effective ensemble size 1.405→1.572; persistent singleton rate .095→.344.

This is a **descriptive mechanism comparison**, not compute-matched causal evidence.

### 5.2 Module2: S1 → S2

| Seed | Reinforcement rate S1→S2 | Orphan conversion rate S1→S2 | Median conversion latency | Persistent singleton rate | Vote yield/member gain | Final gap |
|---:|---:|---:|---:|---:|---:|---:|
| 48 | .588→.800 | .588→.667 | 3→2 | .412→.267 | .450→.500 | 8→7 |
| 49 | .591→.706 | .500→.647 | 3→2 | .455→.294 | .240→.308 | 15→9 |
| 50 | .833→.688 | .833→.563 | 3.5→2 | .167→.250 | .500→.368 | 7→8 |

S2 improved the intended conversion signature in seeds 48 and 49, but not seed50. Median conversion latency improved in all three seeds, yet seed50 converted a smaller fraction and left a larger persistent fraction. Final S2 vote was 58 in all three seeds, but that aggregate does not erase the trajectory inconsistency.

The Module2 verdict is **MIXED / PARTIALLY SUPPORTED**: improvement in 2/3 seeds is useful development evidence, but it is not stable evidence.

## 6. Orphan Lifetime and Exact Vote Boundary

S1 created 51 orphan episodes across the three seeds. Their final episode fates were:

- 31 rescued to a correct vote;
- 19 singleton until run end;
- 1 lost coverage again;
- 0 reinforced but still vote-wrong.

S2 created 48 episodes:

- 30 rescued to a correct vote;
- 13 singleton until run end;
- 2 lost coverage again;
- 3 reinforced but still vote-wrong.

All repair distances use exact plurality replay, not a `G>=3` majority approximation. The state table separately records fragmented coverage, singleton-wrong, one-repair-away, and repair-distance-2+ stocks. Event-level timing and terminal fate are in `orphan_creation_events.csv`; descriptive survival values are in `orphan_survival.csv`.

## 7. Cross-Member Responsibility Revisit

Branch-assignment evidence shows substantial propagation:

| Setting | Orphans | Revisited by another member | Rate |
|---|---:|---:|---:|
| S1 | 51 | 48 | .941 |
| S2 | 48 | 42 | .875 |

For S1, this means portfolio/branch assignment revisit; explicit responsibility-conditioned context is not applicable. For S2, 24/48 episodes had an explicit later different-member conditioned-context hit. Among S2 episodes, 25 conversions occurred after an observed cross-member branch revisit.

Source semantics and trajectory evidence agree: routing is unique at a state, but eligibility is recalculated for currently wrong members after a team transition. Once the discovering member becomes correct, other wrong members can still be eligible and receive the residual. Thus unique ownership is **not** equivalent to permanent residual removal.

The supported diagnosis is narrower: branch-level revisit is sufficient, while explicit context propagation is incomplete and downstream candidate acceptance remains a bottleneck.

## 8. Late-Stage Plateau Diagnosis

Terminal no-commit suffixes were:

- Seed48: S1 last accepted at update17 (14 attempts remained); S2 at update22 (9 remained).
- Seed49: S1 at update15 (16 remained); S2 at update24 (7 remained).
- Seed50: S1 accepted at update31, so there is no observable terminal plateau; S2 last accepted at update19 (12 remained).

Across observable S1 terminal plateaus, 50 of 60 selected branches ended at generation/critic scarcity and 10 at common-safe conflict. Across S2 terminal plateaus, 34 of 56 ended at common-safe conflict, 21 at generation/critic scarcity, and 1 at no target-or-vote progress. Seed50 S2 was the exception where generation/critic scarcity (13 branches) exceeded common-safe conflict (10).

The W1 repairability discount visibly reduced expected update values after repeated failures, but did not guarantee conversion-oriented feasibility. This distinguishes selector mechanics from overall Module1/2 efficacy.

Important limitation: rejection reasons are candidate-level and may overlap. Existing artifacts do not persist a rejected candidate's per-example repair-distance vector, so a specific rejection reason cannot be uniquely assigned to an individual orphan. That field is marked `UNSUPPORTED_BY_EXISTING_ARTIFACTS` rather than inferred.

## 9. Research Questions

### RQ1 — Did Module1 produce stronger specialization than generic evolution?

**YES, descriptively.** All three seeds show more discoveries, higher oracle coverage, lower correctness correlation, and higher effective ensemble size for S1 than S0.

### RQ2 — Was that specialization useful consensus or fragmented oracle coverage?

**Both, but disproportionately fragmented.** S1 gained some votes, but the oracle-vote gap and persistent singleton rate rose in every seed; conversion efficiency improved in only one seed.

### RQ3 — What happened to Module1's newly discovered residuals?

Across 51 S1 orphan episodes: 31 converted, 19 remained singleton through the end, and 1 lost coverage. Per-seed exact records are in `orphan_creation_events.csv`.

### RQ4 — Did Module2 stably improve fragmented-coverage conversion?

**MIXED / NO stable improvement.** Seeds 48 and 49 improved; seed50 worsened reinforcement rate, conversion rate, persistent rate, vote yield, and final gap relative to S1.

### RQ5 — What is Module2's main failure layer?

**Common-safe candidate conflict, with a seed-specific generation/critic component.** The dominant aggregate S2 plateau layer was common-safe conflict; seed50 was generation/critic dominated. Explicit conditioned-context propagation reached only half of S2 orphan episodes.

### RQ6 — Does evidence support changing unique ownership into discover-and-propagate-until-boundary?

**INSUFFICIENT EVIDENCE for committing to a new method.** Studying stronger explicit propagation is motivated, but the premise that unique ownership currently prevents later cross-member assignment is contradicted. A fixed-trajectory ablation should separate context exposure from candidate feasibility before any v16 implementation.

## 10. Verdict and Design Implications

- `MODULE1_SPECIALIZATION_SIGNATURE = STRONG`
- `COVERAGE_FRAGMENTATION = SUPPORTED`
- `MODULE2_CONSENSUS_CONVERSION = MIXED`
- `CROSS_MEMBER_REVISIT = SUFFICIENT_AT_BRANCH_LEVEL; INCOMPLETE_AT_EXPLICIT_S2_CONTEXT_LEVEL`
- `PRIMARY_PLATEAU_CAUSE = S1_GENERATION_OR_CRITIC; S2_COMMON_SAFE_CONFLICT_WITH_SEED50_MIX`
- `NEXT_METHOD_DIRECTION_SUPPORT = INSUFFICIENT_EVIDENCE`

The clean next step is offline fixed-trajectory isolation of two separate questions: whether explicit different-member context exposure can be increased, and whether already-generated candidates fail because they repair the wrong residual geometry or because common-safe objectives conflict. This report does not define, tune, or implement a new method.

## 11. Limitations

- Three development seeds support only per-seed directions and descriptive means; no p-values, confidence intervals, or significance claims are made.
- S0→S1 is not compute-matched causal evidence.
- Candidate-level per-example rejection attribution and rejected-candidate repair-distance changes are unsupported by stored artifacts.
- Context hashes establish question-level exposure, not semantic equivalence of prompt patterns.
- This audit makes no test-set, efficacy, generalization, or held-out claim.
