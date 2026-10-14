# V2.12 exhaustive correctness-count responsibility

Current method: `unified_team_prompt_search_v2_12_count_responsibility_v1`.
The user explicitly replaced the previous Vote-counterfactual responsibility
definition with correctness counts only. This is a scientific amendment, not a
claim that the historical D/N predicates always represented these counts.
Historical V2.10/V2.11 contracts, reports and trajectories remain immutable.

For each Optimize example, read the five benchmark Boolean member correctness
results. A correct member receives zero repair responsibility. Every wrong member
receives exactly one label according to the total number of correct members:

| Correct members | Wrong members | Category | Weight | Serialized label |
|---:|---:|---|---:|---|
| 0 | 5 | C | 1 | coverage |
| 1 | 4 | N | 2 | near_margin |
| 2 | 3 | D | 4 | direct_flip |
| 3 | 2 | R3 | 1 | r3 |
| 4 | 1 | R4 | 0.5 | r4 |
| 5 | 0 | No member error | 0 | No label |

The first three serialized names and count-field names are retained as aliases
to avoid unrelated interface refactoring. They do NOT assert an actual Vote flip,
near-margin repair or absence of a correct peer. The new closed classification
identity and immutable policy explicitly record their count meanings. All wrong
members on the same example receive the same category, including invalid outputs
which the unchanged benchmark scores incorrect. No residual error gets zero score.

Classification must not read answers, equivalence classes, Vote correctness,
margin, gold-replacement outcomes or an eligibility assignment. Invalid five-member
correctness vectors and repeated example identities fail closed. Existing native
Vote and counterfactual diagnostics can still be computed for evaluation and group
evidence; they cannot gate classification, WHO responsibility or Pattern score.
The retained `assigned` observation key now lists all member errors and is not a
lexicographic selection filter. Canonical group evidence verifies the same labels.

Member raw score is R = 4D + 2N + C + R3 + 0.5R4. WHO alone applies the existing
R/(1+failure_count) discount and deterministic tie rule. After unchanged semantic
Gradient clustering, Pattern score sums the same labels' weights once per unique
wrong support ID, without WHO discount. Representative supports are then sampled
by the existing policy. No new normalization, Preservation constraint or category
is introduced. Production evidence requires one label for each wrong member row.

Three independent with-replacement Parent draws, one-step Reflection, shared
Layer2 repair direction, max-three pools, Fitness, scoped Memory, Group diagnosis,
aggregation, Full/Shadow admission, final argmax-team evaluation, Solver/scoring,
split isolation and stopping are unchanged. Count-only responsibility is not a
measurement of actual marginal Vote contribution: a two-correct example can have
a correct, tied or incorrect Vote depending on the wrong answer partition.

New method, classification, WHO/Pattern, binding, composition, cache and checkpoint
identities prevent old-source execution and scope/checkpoint reuse. The MATH
protocol changes its responsibility policy identity only; scoring, parsing and
aggregation are unchanged. All behavior policies enter method, manifest, binding,
execution scope and source hashes. Preparation remains DRAFT and unauthorized.

No additional fixed generation or Solver evaluation budget is added. Gradient
already diagnoses all selected-member errors. WHO/Pattern choices and realized
input lengths, retries and Tokens can change; their direction and scientific
effects are unknown without a fresh frozen real experiment. Fixed historical
recounting can identify changed labels and rankings, not new sequential evolution,
candidate quality, Commit/Vote gains or generalization.
