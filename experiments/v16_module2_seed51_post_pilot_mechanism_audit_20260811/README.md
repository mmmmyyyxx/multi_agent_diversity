# Seed51 post-Pilot Module2 mechanism audit

This is a zero-API, train-only, post-Pilot analysis of the immutable Seed51
C0/C2/C3 artifacts. The canonical re-audit passed after an auditor-only schema
reconciliation at commit `7347a6bcc2c307a13af6c35e111b492b9282ef32`.
The executed runtime remains commit
`1f3947c6e69b26666614d8be82607175c76048ee`. All 128 files under the original
Pilot root had identical SHA256 values before and after re-audit.

## Canonical re-audit

The gate is **PASS** with zero blockers. It reconciled 32 context rows to the
persisted `candidate_decisions.jsonl` parent hash by update index, without
modifying or backfilling any run artifact. W1, common-safe, max-one-commit,
vote-correct propagation, context serialization, and C2/C3 same-parent
membership mismatch counts are all zero. Validation and test counts are zero.

## 1. Why C3 still loses P1 capability

Ten of 23 evaluated C3 candidates lost at least one P1 capability, for 18 P1
losses in total. They concentrate in updates 2, 3, 4, 5, and 7.

- Four of the ten combined repair gain with a P1 loss. The other six damaged
  P1 without any repair gain.
- Nine of ten were rejected by common-safe. Six are plain F regressions with
  no repair gain; two repair candidates had target gain -7 and only exchanged
  three vote gains for three vote losses.
- One P1-loss candidate was feasible and committed: update 3, target 0. It had
  target gain 0, seven vote gains, three vote losses, net vote +4, two R1
  repairs, and three P1 losses. This is legal because P1 is generation guidance,
  not an acceptance guard.
- Across all 13 C3 repair-gain candidates, six were preservation-safe and seven
  had P1/P2 collateral. Only four of those seven collateral cases involved P1;
  the remaining three involved P2 only.

The dominant failure is therefore not a hidden acceptance violation. It is a
generation problem with two forms: broad regressions that repair nothing, and
occasionally high-net-vote trades that sacrifice vote-critical competence.

## 2. Why C2 generated only ten valid candidates

The collapse occurs before Student generation, at the Critic semantic gate.

- C2: 24/30 Critic responses rejected; ten of sixteen branches exhausted the
  Critic gate; only six branches reached Student.
- C3: 13/25 rejected; four of sixteen branches exhausted; twelve reached
  Student.
- C0: 10/23 rejected; three of sixteen branches exhausted; thirteen reached
  Student.
- C2 had 23 `preservation_or_output_risk` rejections, versus 13 for C3. It had
  no Teacher/Critic JSON failure, no truncation, no Student terminal schema
  failure, and no infrastructure failure.
- Once Student was reached, C2 returned 12 raw candidates and ten valid ones;
  C3 returned 24 and 23. Thus Student validity is secondary, not the main cause.
- C2 contexts were shorter than C3 (mean 4,496 versus 5,146 characters; maxima
  5,272 versus 6,308), so output collapse is not explained by a context-length
  overflow.

The evidence supports a semantic explanation: the unlabeled C2 preservation
list frequently caused the Critic to reject the Teacher plan as a preservation
or output risk. C3 metadata reduced, but did not eliminate, that bottleneck.

## 3. Same-parent attribution coverage

There is exactly one clean tri-arm parent state: update 0, with the same parent
team hash and selected targets 2 and 4. C2 and C3 Repair/Preservation membership
is identical for both targets.

- Target 2: C0 and C2 exhausted the Critic gate; C3 produced two B candidates,
  both feasible.
- Target 4: C0 and C2 produced two F candidates, none feasible; C3 exhausted
  the Critic gate.
- Aggregated over these two matched branches, C2 produced two valid and zero
  feasible candidates; C3 produced two valid and two feasible candidates.

This is useful but insufficient for a clean general metadata claim: it is one
parent state and two branches, and the branch on which generation succeeds
changes between variants. Later arm-wide differences include trajectory
divergence and cannot be interpreted as a fixed-context metadata effect.

## Decision

The canonical Seed51 gate is now PASS, but canonical-v16 promotion remains
unjustified. C2 is negative relative to C0. C3 repairs much of C2's semantic
gate collapse, yet remains worse than C0 on F and feasible rate and still has
material P1 collateral. If further evidence is desired, the clean next
experiment is a preregistered, very small fixed-parent generation probe with no
commits—not Seed52 and not a final test. No method change is made here.
