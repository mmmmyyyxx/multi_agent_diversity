# Endpoint Structural Identifiability Preflight

Run this zero-API preflight before an experiment spends provider budget to ask
whether changing one member can change a five-member plurality result. It is an
experiment-design check; it does not change target scheduling, candidate
acceptance, voting, or any runtime method.

For every frozen evaluation row and possible target member:

1. hold the other four recorded outputs fixed;
2. enumerate every legal target answer plus invalid/abstain;
3. aggregate each possibility with the frozen equal-weight plurality rule and
   tie-as-abstain behavior;
4. count the row when at least two target outputs yield different plurality
   outcomes.

Record:

```text
P_i = number of plurality-pivotal-capable rows for member i
```

A single-member transfer endpoint is structurally identifiable only when:

```text
sum_i P_i > 0
```

The preregistration and handoff must contain the per-member counts, total,
profile provenance and hashes, legal-output-domain provenance, tie rule, and a
machine-readable audit artifact. A zero total fails closed before API use for
that endpoint. It does not show that local repair, team transfer in another
state, or vote-neutral symmetry breaking is ineffective.

Use
`multi_dataset_diverse_rl.evaluation.endpoint_identifiability.endpoint_structural_identifiability`
for the enumeration. Do not substitute prompt-hash diversity, member-accuracy
diversity, or a hand-written margin shortcut for this replay.
