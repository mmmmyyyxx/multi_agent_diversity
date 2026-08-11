# v16 Module2 Candidate Design + Fixed-Trajectory Offline Isolation

## 1. Executive Summary

The study passes. C0–C3 were frozen before analysis, including budgets, set construction, F subtypes, proxy definitions, and recommendation rules. The frozen files retained their recorded SHA256 throughout replay.

Exact context reconstruction shows that deterministic boundary-first repair sets substantially close the known propagation gap in every seed: orphan event context coverage rises from 24/48 under C0 to 42/48 under C1–C3, while branch-assigned-to-context retention rises from 46.6% to 96.6%. No vote-correct residual enters a proposed repair set and cross-branch repair duplication is zero.

Preservation is useful but must be tier-aware. The full P1+P2 set overlaps 71.9% of D/F harmful candidates, but also 48.5% of A/B candidates. P1 vote-critical items alone are much more discriminative: 15.2% A/B, 100% D, and 63.0% F. Seven of the eight previously identified local-repair/global-failure candidates lose at least one frozen P1/P2 item.

The recommended design is **C3 Coalition-Aware Repair + Preservation**, with **MEDIUM** confidence. C3 has the same repair and preservation membership as C2, but explicitly communicates exact repair distance, boundary role, and preservation tier—the distinctions that the fixed-pool replay found informative. This is a design recommendation only. Whether C3 generates better candidates is `REQUIRES_PILOT`.

## 2. Evidence Basis

- Method semantics commit: `c705eedb2959c3ad1349f5d6c52ffed64bca90ae`
- Historical execution commit: `b7936ae2f16d8907f0ffdf161dc8991368abeed8`
- Current analysis HEAD: `358c33cff5afa5ff78b21706e96aab4db36c3adc`
- Seeds: 48, 49, 50
- Setting: historical S2 train trajectories only
- Historical candidates: 247, replay mismatches 0
- API/model/training/validation/test calls: 0

The first audit established specialization and fragmentation. The second established a 42→24 branch/context loss and rejected geometry dominated by F, then D, with E rare. Those facts are treated as constraints rather than rediscovered.

## 3. Frozen v15 Failure Mode

```text
Module1 discovers complementary coverage
  → another wrong member is eligible and eventually selected
  → 42/48 orphan episodes reach another branch
  → only 24/48 enter explicit conditioned context
  → local repair sometimes occurs
  → global target/vote collateral often destroys feasibility
```

Acceptance remains unchanged. The study does not soften target or vote non-regression.

## 4. Design Principles

### 4.1 Preserve specialization

Repair sets are constructed only for the selected historical branch target. They do not alter Module1 routing or target selection.

### 4.2 Minimal necessary redundancy

Only target-wrong, vote-wrong, uniquely branch-assigned residuals are eligible. Vote-correct residuals added by coalition propagation are exactly zero. Repair items never duplicate across the two branches of an update.

### 4.3 Vote-boundary responsibility

Exact repository plurality and enumerated repair distance rank `r=1` first, then unresolved singleton coverage, then `r=2`, then remaining responsibility.

### 4.4 Global competence preservation

The target-invalid counterfactual is supported by repository validity-aware plurality. P1 marks correct→incorrect vote flips; P2 marks reduced correct plurality margin; P3 is stable competence. The six-item budget was always filled by P1/P2 in this trajectory, so P3 never entered the frozen sets.

## 5. Candidate Variants

- C0: exact historical v15 context.
- C1: six-item maximum boundary-first repair set.
- C2: C1 plus six-item maximum P1/P2/P3 preservation set.
- C3: identical C2 membership plus compact `G`, `r`, boundary class, target role, and preservation-tier metadata.

Synthetic prompt prototypes use placeholders only and are stored in `design_variants.json`; no train text appears in this directory.

## 6. Exact Context Construction Analysis

| Seed | C0 event coverage | C1–C3 event coverage | C0 assigned→context | C1–C3 assigned→context |
|---:|---:|---:|---:|---:|
| 48 | 7/15 | 12/15 | 31/51 | 49/51 |
| 49 | 9/17 | 16/17 | 26/77 | 72/77 |
| 50 | 8/16 | 14/16 | 38/76 | 76/76 |
| All | 24/48 | 42/48 | 95/204 | 197/204 |

Eligible-opportunity context rate rises consistently:

- Seed48: 5.92%→9.35%;
- Seed49: 4.01%→11.11%;
- Seed50: 6.44%→12.88%;
- pooled descriptive: 5.39%→11.18%.

Only seven assigned opportunities remain excluded under the fixed six-item budget: two in seed48 and five in seed49. Their explicit exclusion reason is budget/priority, not selector or routing failure.

## 7. Repair-Set Coverage

Across 192 selected branches, the frozen sets contain 722 repair items:

- R1 one-repair-away: 244 (33.8%);
- R2 singleton fragmented: 138 (19.1%);
- R3: 0;
- R4 other assigned residuals: 340 (47.1%).

R1+R2 comprise 52.9%. Thus boundary-first construction produces a **strong exposure recovery but mixed boundary precision**: it prioritizes the intended boundary/orphan stock, yet nearly half of included items are lower-priority responsibility because capacity remains after higher tiers are exhausted.

R1 opportunity exposure rises from 16.6% under C0 to 25.9% under C1–C3. This is meaningful but not equivalent to making all context items `r=1`.

## 8. Preservation-Set Analysis

The 192 branch states produce 1,152 frozen preservation items: 923 P1 and 229 P2; P3 receives no slot because higher tiers fill the six-item budget.

### P1-only discrimination

| Geometry | Any P1 loss |
|---|---:|
| A/B useful | 5/33 = 15.2% |
| D target+/vote− | 30/30 = 100% |
| F | 109/173 = 63.0% |
| D/F | 139/203 = 68.5% |

### Full P1+P2 discrimination

| Seed | A/B violation | D violation | F violation | D/F capture |
|---:|---:|---:|---:|---:|
| 48 | 60.0% | 100% | 62.9% | 68.9% |
| 49 | 41.7% | 100% | 79.4% | 82.4% |
| 50 | 45.5% | 100% | 56.3% | 61.8% |
| All | 48.5% | 100% | 67.1% | 71.9% |

The direction is consistent in all seeds: D/F capture exceeds A/B overlap. But the full set has nontrivial A/B overlap, so preservation must remain soft guidance rather than a hard rejection rule. P1's much cleaner separation makes explicit tier labels useful.

## 9. Historical Candidate Alignment

These are fixed-pool proxies over C0-generated candidates, not variant efficacy:

| Variant | Aligned candidates | Rate |
|---|---:|---:|
| C0 | 68/247 | 27.5% |
| C1 | 100/247 | 40.5% |
| C2 | 20/247 | 8.1% |
| C3 | 20/247 | 8.1% |

C1's broader deterministic repair set finds more historical local alignment. C2/C3 require zero preservation loss and are intentionally stringent; their low retrospective rate says the old pool rarely combines repair and preservation, not that a new context cannot generate such candidates.

## 10. F-Type Decomposition

The frozen mutually exclusive decomposition of 173 F candidates is:

- `F_TARGET_DEGRADATION`: 86 (49.7%);
- `F_LOCAL_GAIN_GLOBAL_COLLATERAL`: 64 (37.0%);
- `F_LOCAL_NO_PROGRESS`: 23 (13.3%);
- `F_OTHER`: 0.

This sharpens the generator diagnosis. Only 13.3% are pure local no-progress; most either broadly degrade the target or repair a selected responsibility while losing globally.

## 11. Eight Local-Repair / Global-Failure Cases

All eight prior cases were reproduced exactly and remain anonymized by hashes. Seven of eight lose at least one frozen P1/P2 item; together the eight have 14 P1/P2 losses. The remaining case has no frozen preservation loss under the six-item budget.

Per-case fields are in `local_repair_rejected_cases.csv`. The correct interpretation is that C2/C3 **would expose relevant preservation regions** for seven cases. It is not evidence that the prompt would prevent those regressions.

## 12. Fixed-Pool Oracle Availability

At each historical branch/update, the oracle asks whether any C0-generated common-safe candidate both repairs the variant set and, for C2/C3, has zero P1/P2 loss.

| Variant | Seed48 | Seed49 | Seed50 | All |
|---|---:|---:|---:|---:|
| C0 | 5/49 | 6/50 | 4/38 | 15/137 |
| C1 | 7/49 | 7/50 | 4/38 | 18/137 |
| C2 | 3/49 | 4/50 | 1/38 | 8/137 |
| C3 | 3/49 | 4/50 | 1/38 | 8/137 |

Ideal repair+preservation candidates are nonzero in all seeds but scarce. This supports a small generation Pilot while showing that new generation—not only re-ranking the existing pool—is necessary.

## 13. Context Complexity

| Variant | Mean items | P95 items | Max items | Mean placeholder chars | P95 chars |
|---|---:|---:|---:|---:|---:|
| C0 | 1.84 | 2 | 2 | 2,198 historical full serialization | 2,544 |
| C1 | 3.76 | 6 | 6 | 81 | 115 |
| C2 | 9.76 | 12 | 12 | 214 | 248 |
| C3 | 9.76 | 12 | 12 | 541 | 734 |

C0 full serialized characters are not directly comparable to synthetic placeholder strings. Within the comparable prototypes, C3 is about 2.5× C2 in characters but remains bounded below 742 placeholder characters. C2/C3 have 323 cross-branch duplicate preservation occurrences; repair duplication remains zero. The duplicates reflect shared capabilities worth preserving, not propagation of already-correct residuals, but they create capacity pressure.

## 14. Three-Seed Consistency

The following directions hold in 3/3 seeds:

- C1–C3 improve orphan event exposure over C0;
- C1–C3 improve assigned→context retention;
- full preservation D/F capture exceeds A/B overlap;
- D candidates have 100% P1/P2 preservation violation;
- C2/C3 fixed-pool ideal availability is nonzero but lower than C1;
- C3 and C2 have identical membership/alignment.

Magnitude varies, so the design recommendation is medium rather than high confidence.

## 15. Design Questions

### DQ1 — Can deterministic construction reduce branch-assigned non-exposure?

**Yes.** Retention rises from 95/204 to 197/204 and event coverage from 24/48 to 42/48.

### DQ2 — Is boundary-first prioritization actually boundary-focused?

**Mixed but useful.** R1+R2 are 52.9% of items and R1 exposure improves, while 47.1% are lower-priority R4 items.

### DQ3 — Does Preservation capture historical collateral damage?

**Yes, with nontrivial overlap.** It captures 100% of D, 67.1% of F, and seven of eight focal failures, but also 48.5% of A/B.

### DQ4 — Do P1/P2 discriminate useful from harmful candidates?

**P1 strongly discriminates D and moderately discriminates F; P2 dilutes specificity.** P1: 15.2% A/B versus 100% D and 63.0% F. Full P1+P2: 48.5% A/B versus 71.9% D/F.

### DQ5 — How many focal losses fall in C2/C3 preservation?

**Seven of eight candidates**, covering 14 P1/P2 losses.

### DQ6 — Does C3 add enough information over C2?

**Design-level yes, efficacy unknown.** P1 versus P2 and exact boundary role are materially different signals. C3 explicitly conveys them; C2 only supplies an unlabelled preservation list. The extra cost is bounded but material. Whether the model uses the labels is `REQUIRES_PILOT`.

### DQ7 — Is there realistic fixed-pool space for better generation?

**Limited but nonzero.** C2/C3 ideal candidates exist in every seed but only 8/137 branch updates. Re-ranking alone is insufficient; new generation behavior is necessary.

## 16. Recommended Module2 Design

Recommend `C3_COALITION_AWARE_PRESERVATION`, confidence `MEDIUM`.

Conceptually:

```text
Boundary-Aware Repair with Competence Preservation
  = compact boundary-first repair responsibilities
  + explicitly tiered vote-critical preservation responsibilities
```

This completes the research chain:

```text
Discover → Reinforce → Preserve → Consensus
```

C3 is selected over C2 because the replay demonstrates that the P1/P2 distinction itself matters; it is not selected because “more metadata is better.” If a Pilot shows no generation advantage, the simplicity fallback is C2.

## 17. Evidence Levels

### Level A — EXACT

Context membership, exact repair distance, preservation membership, historical candidate deltas, candidate geometry, capacity, and P1/P2 overlap.

### Level B — FIXED-POOL RETROSPECTIVE

Historical candidate alignment, preservation discrimination, eight focal cases, F subtypes, and oracle availability. All candidates came from C0.

### Level C — REQUIRES PILOT

- C3 generates better candidates;
- C3 improves team vote;
- C3 beats C2;
- any generalization claim.

## 18. Minimal Pilot Proposal — Not Executed

After separate explicit API authorization only:

```text
one fresh seed
qwen3-14b
thinking=false
train-only
no validation
no final test
small fixed update budget
C0 versus C3 only
same Module1, candidate budget, and common-safe acceptance
```

No Pilot or runtime implementation was performed in this task.
