# V2.10 exclusive weighted error responsibility

The sole current method is `unified_team_prompt_search_v2_10_weighted_responsibility_v1`.
This amendment supersedes V2.9 member and Pattern raw responsibility, not its
Parent Population search, scoring, Gradient generation, semantic clustering,
Reflection, Memory, Full/Shadow admission or stopping. Historical identities,
manifests, source and results retain their original meaning.

Every target-member Optimize error has at most one positive responsibility label:

| Label | Predicate | Weight |
|---|---|---|
| D / direct_flip | Existing counterfactually assigned team-wrong error whose single-member gold replacement makes Vote correct | 4 |
| N / near_margin | Existing assigned team-wrong error with no Vote flip, positive margin gain and repaired margin exactly zero | 2 |
| C / coverage | Existing assigned error with no correct peer, after the D/N predicates fail | 1 |
| R3 / r3 | Team Vote correct, target wrong, exactly three of five members correct | 1 |
| R4 / r4 | Team Vote correct, target wrong, exactly four of five members correct | 0.5 |

The user explicitly clarified R3/R4 on this task. They are not historical
Module2 repair-distance tiers. D/N keep actual gold-replacement Vote and margin
counterfactuals with the same abstaining tie semantics; correct-member count does
not replace those predicates. Preserve the existing lexicographic
(vote_flip_gain, margin_gain) assignment among wrong members on team-wrong rows.
Coverage, invalid outputs and fixed peer validity use the same existing algebra.
The D-before-N-before-C precedence preserves the old exclusive telemetry lane.
R3/R4 are computed separately on team-correct rows, which the old assignment
builder skips. They enter actual member diagnosis and the shared per-example
labels delivered to Gradient/Pattern, not just new aggregate fields.

R = 4D + 2N + C + R3 + 0.5R4. Counts are nonnegative integers; scores are floating
values (half-unit precision). WHO uses R/(1+member failure count). Existing eligible
member, tie and seeded all-zero fallback rules remain. Pattern scores sum these
same exclusive weights across every support ID after semantic partition/merge;
they have no failure discount. Highest positive score wins, with the existing
canonical identity tie rule and seeded all-zero fallback. The descriptive primary
lane is telemetry only and never replaces the summed member responsibility.

Uncovered errors have zero weight and remain eligible under existing fallback.
Report their exact counts and reasons: team-correct with another correct-member
count, team-wrong not assigned by the existing counterfactual rule, and assigned
team-wrong without D/N/C. Correct member observations have no repair weight.
No additional class, inferred weight, normalization or Preservation constraint
is introduced. Duplicate IDs, overlapping labels and malformed counts fail closed.

The complete immutable policy enters method mechanisms, binding, single-use
execution scope, manifest schema and scientific hash. New method, responsibility,
WHO/Pattern scoring, composition and cache identities prevent reuse of V2.9
checkpoints or authorizations. V2.9 Parent Population component policies remain
identical: three frozen with-replacement draws, max three unique Parents per
member, separate retention/deployment, and one jointly evaluated final argmax team.

All wrong examples were already Gradient-eligible, including R3/R4. The change
adds no fixed model calls, generations, panels or evaluator requests. Selection
can change the number/type of diagnosed errors and realized Token costs; those
effects and Team Vote gains remain unknown without a new frozen real experiment.
Offline historical comparisons hold prior states, failure counts and available
partitions fixed. They are not a new sequential trajectory or causal efficacy
comparison; unseen member-specific Gradients/Patterns cannot be inferred.
