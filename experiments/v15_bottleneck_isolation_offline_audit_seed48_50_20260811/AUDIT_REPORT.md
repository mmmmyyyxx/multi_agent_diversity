# v15 Bottleneck Isolation Offline Audit

## Executive Summary

The audit passes and rejected-candidate replay is fully supported. All 247 evaluated S2 candidates across seeds 48–50 had complete 75-question observations in their original setting-local SQLite cache. The caches were opened in read-only URI mode; exact repository plurality replay reproduced every stored candidate vote vector, G vector, target gain, vote gain count, and vote loss count with zero mismatch.

The two bottlenecks are now separable:

1. **Propagation reaches eligibility and target selection, but loses half the orphan episodes before explicit context.** All 48 orphan episodes had at least one later eligible wrong member and all 48 eventually had such a member selected. Forty-two reached another member's branch assignment, but only 24 entered that member's explicit conditioned context.
2. **Once candidates exist, candidate quality—not an overly strict target guard—is the dominant rejection geometry.** Of 214 rejected candidates, 173 (80.8%) had no positive joint target/vote geometry, 30 (14.0%) improved the target but harmed the vote, and only 11 (5.1%) improved the vote while regressing the target.

Therefore the evidence does not justify relaxing common-safe acceptance. The most defensible design direction is to improve explicit context construction and globally coalition-aware candidate generation, while preserving current acceptance until stronger evidence appears.

## Scope and Provenance

- Git HEAD: `d72e1515556101f5af211ebb56b2f375ea8825ca`
- Seeds: 48, 49, 50
- Setting: S2 `shared_responsibility_conditioned_dual_target`
- Train trajectory only
- API/model/training/test calls: 0
- Test artifacts consumed: false
- Method source changed: false
- Images generated: false

`audit_manifest.json` lists every consumed artifact and SHA256. The raw caches are not copied or modified.

## 1. Opportunity-Normalized Propagation Funnel

### Event-level strict funnel

| Stage | Seed48 | Seed49 | Seed50 | All |
|---|---:|---:|---:|---:|
| Orphan created | 15 | 17 | 16 | 48 |
| Another wrong member eligible | 15 | 17 | 16 | 48 |
| Such a member selected | 15 | 17 | 16 | 48 |
| Residual assigned to its branch | 12 | 16 | 14 | 42 |
| Explicit conditioned context hit | 7 | 9 | 8 | 24 |
| Valid/evaluated candidate | 7 | 9 | 8 | 24 |
| Common-safe feasible candidate | 5 | 6 | 4 | 15 |
| Accepted candidate on that path | 5 | 6 | 4 | 15 |
| Same strict-prefix episode eventually converted | 2 | 5 | 4 | 11 |

The independent eventual conversion count is 30/48. It is intentionally reported separately: 19 conversions occurred without satisfying the full explicit-context→feasible→accepted prefix for that orphan, so treating conversion as a simple nested final stage would be false.

### Opportunity-level normalization

Across the active orphan lifetimes there were 1,762 `(orphan, raw update, eligible other agent)` opportunities:

| Scope | Eligible opportunities | Selected | Branch assigned | Explicit context | Valid candidate | Feasible | Accepted |
|---|---:|---:|---:|---:|---:|---:|---:|
| Seed48 | 524 | 209 | 51 | 31 | 22 | 8 | 8 |
| Seed49 | 648 | 300 | 77 | 26 | 21 | 7 | 7 |
| Seed50 | 590 | 240 | 76 | 38 | 23 | 7 | 7 |
| All | 1,762 | 749 | 204 | 95 | 66 | 22 | 22 |

Thus:

- `P(context hit | eligible agent-update opportunity) = 95/1762 = 5.39%`;
- `P(context hit | assigned agent-update opportunity) = 95/204 = 46.57%`;
- event-level assignment→context retention is `24/42 = 57.14%`;
- event-level context→feasible retention is `15/24 = 62.5%`;
- opportunity-level valid-candidate→feasible retention is `22/66 = 33.3%`.

Interpretation:

- Eligibility is not the problem: 48/48 events had another eligible wrong member.
- Long-run selector coverage is not the problem: 48/48 eventually selected one.
- Branch routing is imperfect but relatively high: 42/48.
- The first major event-level loss is branch assignment→explicit context: 18 of 42 branch-revisited episodes never reached explicit conditioned context.
- The second major loss is candidate feasibility after context.

## 2. Rejected-Candidate Counterfactual Replay

### Recovery status

| Seed | Evaluated candidates | Unique prompts | Missing prompt-question observations | Replay mismatches |
|---:|---:|---:|---:|---:|
| 48 | 87 | 86 | 0 | 0 |
| 49 | 91 | 90 | 0 | 0 |
| 50 | 69 | 68 | 0 | 0 |

The repeated prompt hashes are legitimate reuse. Every unique prompt had exactly 75 ready train observations. `REJECTED_CANDIDATE_COUNTERFACTUAL_REPLAY = SUPPORTED_COMPLETE`.

For each candidate/question pair, the audit rebuilt the five-member team after replacing only the target member, invoked repository plurality, and computed exact `ΔG`, `ΔVote`, and enumerated `Δrepair_distance`. No answer content is persisted.

## 3. Common-Safe Conflict Geometry

### All evaluated candidates

| Type | Definition | Count | Share |
|---|---|---:|---:|
| A | target+ / vote+ | 11 | 4.5% |
| B | target+ / vote0 | 22 | 8.9% |
| C | target0 / vote+ | 0 | 0.0% |
| D | target+ / vote− | 30 | 12.1% |
| E | target− / vote+ | 11 | 4.5% |
| F | no positive joint value | 173 | 70.0% |

All 33 A/B candidates passed common-safe feasibility. Twenty-five became the global max-one commit; eight lost cross-branch competition. There were no C candidates.

### Rejected candidates only

There were 214 rejected candidates:

| Type | Count | Rejected share | Design meaning |
|---|---:|---:|---|
| D: target+ / vote− | 30 | 14.0% | Local specialization damages coalition vote |
| E: target− / vote+ | 11 | 5.1% | Team-helpful candidate blocked by target non-regression |
| F: no joint value | 173 | 80.8% | Candidate quality/no-progress dominates |

The exact rejected sign matrix is:

- target− / vote−: 92;
- target− / vote0: 56;
- target+ / vote−: 30;
- target0 / vote0: 15;
- target− / vote+: 11;
- target0 / vote−: 10.

This rules out “acceptance is mainly discarding team-helpful candidates” as the primary diagnosis. Only 11/214 rejected candidates are type E. Relaxing target non-regression would address a small minority while admitting a new safety tradeoff.

Type D is real but secondary: 30 candidates improved their member while harming team vote. This supports adding coalition/vote-boundary information to generation context, not weakening vote safety.

Type F is decisive: 173 rejected candidates had neither a safe target improvement nor a team improvement. Generator/candidate quality is the dominant failure layer.

## 4. Did Explicit Context Repair the Intended Residual?

Thirty-six evaluated candidates were generated while an active orphan appeared in that branch's explicit context:

- 15 repaired at least one explicit-context residual locally;
- 9 passed common-safe feasibility;
- 8 repaired a context residual locally but were still rejected.

The eight locally repairing but rejected candidates were 5 type F, 2 type D, and 1 type E. Together they made 11 context-residual repairs, but also produced 65 target-example losses versus 46 target-example gains and 11 vote losses versus 3 vote gains across the full probe. This is direct evidence that context can induce the intended local change while collateral damage elsewhere destroys global feasibility.

The local-to-global gap, rather than absence of all context signal, is therefore the more precise generation problem.

## 5. Bottleneck Decision Table

| Diagnostic dimension | Observed status | Decision |
|---|---|---|
| Eligible opportunity | High: 48/48 events | Not primary |
| Other member eventually selected | 48/48 | Not primary at event level |
| Branch revisit | 42/48 | Secondary routing loss |
| Explicit context | 24/48; 95/1762 opportunities | Material context-construction bottleneck |
| Valid candidate after event-level context | 24/24 | No event-level total generation collapse |
| Feasible after context | 15/24 events; 22/66 opportunities | Material candidate-quality bottleneck |
| Rejected F geometry | 173/214 | Primary generator/global-quality failure |
| Rejected D geometry | 30/214 | Secondary coalition-awareness failure |
| Rejected E geometry | 11/214 | Insufficient evidence to relax target guard |
| Feasible but not committed | 8/33 | Normal max-one competition, not primary |

## 6. Method-Development Verdict

The combined diagnosis is:

```text
Propagation: moderate at branch level, low at explicit opportunity level
Candidate quality: poor globally
Common-safe conflict: real, dominated by F and then D—not E
```

According to the preregistered decision logic, the next layer to study is **Module2 context plus generation**, specifically:

1. raise explicit orphan inclusion after a branch receives the residual;
2. expose vote-boundary/coalition information so local repair does not create global vote loss;
3. improve preservation/global candidate quality;
4. keep current target/vote non-regression while evidence for E-type tradeoffs remains only 11 candidates.

This is a design implication, not a v16 specification. No method was changed or implemented.

## 7. Limitations

- Three development seeds provide descriptive mechanism evidence only.
- Opportunity counts are repeated raw-update opportunities, not independent statistical samples.
- “Valid candidate” means an evaluated candidate persisted after the generation/critic pipeline; zero-candidate branch internals are outside this audit.
- Candidate-level reason labels overlap; geometry is based on exact reconstructed count deltas and is mutually exclusive.
- Context membership proves question-hash exposure, not semantic quality of the natural-language repair plan.
- This audit makes no held-out/test/generalization claim.

## Terminal Summary

```text
AUDIT_STATUS = PASS
API_CALLS = 0
TEST_ARTIFACTS_CONSUMED = false
REJECTED_CANDIDATE_COUNTERFACTUAL_REPLAY = SUPPORTED_COMPLETE
PROPAGATION_PRIMARY_LOSS = BRANCH_ASSIGNMENT_TO_EXPLICIT_CONTEXT
CANDIDATE_PRIMARY_GEOMETRY = F_NO_JOINT_VALUE
COALITION_HARM = SECONDARY_BUT_REAL
TARGET_NONREGRESSION_OVERSTRICTNESS = NOT_PRIMARY
NEXT_LAYER = MODULE2_CONTEXT_AND_GLOBAL_CANDIDATE_QUALITY
METHOD_SOURCE_CHANGED = false
COMMIT_CREATED = false
PUSHED = false
```
