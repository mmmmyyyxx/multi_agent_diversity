# Accepted-Mutation Compositional State Graph v1

## Scope

This is a zero-API retrospective audit of the five frozen mutations from
`accepted_local_mutation_team_transfer_v2`. It asks whether composing existing
local improvements can break the identical-peer plurality lock and whether a
Common-Safe path from the homogeneous baseline must exist.

No Solver, GEPA, Reflection, Validation50, Test50, scheduler, persistent
realizability, or write-back call is permitted.

## Frozen configuration space

Member 0 remains at baseline. Choices for the other members are:

```text
M1: baseline / proposal2
M2: baseline / proposal8
M3: baseline / proposal5
M4: baseline / proposal2 / proposal4
```

The Cartesian product has 24 states, including baseline. Mutation identities,
local telemetry, Full target deltas, loss-case hashes, and baseline profiles are
read without modification from the completed v2 evidence.

## Evidence-availability rule

Exact state metrics require each candidate's 100-row output profile. The audit
must first check whether those profiles or a persistent exact-response cache
exist. Counts, request hashes, correctness totals, and decomposition summaries
must not be treated as answer profiles.

If profiles are absent, the audit must not fabricate exact VoteAcc, oracle
coverage, output diversity, pivotal row identities, or a fully labeled
Common-Safe graph. It instead reports identified intervals and proofs derivable
from frozen counts and loss-case hashes. The required status is:

```text
BOUNDED_AUDIT_COMPLETE_EXACT_GRAPH_NOT_RECONSTRUCTIBLE
```

## Structural identifiability preflight

For each row and target member, fix the other four actual outputs. Enumerate all
legal answer options plus invalid/abstain, apply the repository's frozen
tie-as-abstain plurality rule, and count the row if at least two target outputs
produce different plurality answers.

```text
P_i = count of plurality-pivotal-capable rows for target i
```

Future one-member transfer experiments must report every `P_i` before API use
and require at least `sum_i P_i > 0`. This is experiment-design governance and
does not alter the runtime algorithm.

## Bounded composition rules

The baseline has 60 correct and 40 wrong rows and all five baseline profiles
are identical and valid.

For a state with `k` mutations, each candidate's unique-correct gain set is a
subset of the 40 baseline-wrong rows. When exact gain identities are absent,
the audit uses set-cardinality bounds only. Candidate loss sets are taken from
the frozen collateral-loss hashes after verifying that every assigned
responsibility row was baseline-wrong; therefore those hashes cover every
baseline-correct row lost by that candidate.

For three valid mutations, a baseline-wrong row becomes team-correct exactly
when all three mutations are correct. All three mutations losing a
baseline-correct row is necessary, but their missing wrong-answer clusters
prevent deciding whether the team actually loses that row. The three-way loss
intersection is therefore an upper bound on vote losses. Pair states remain
vote-equal to baseline because three unchanged identical peers retain
plurality.

For four mutations, a baseline-wrong row is guaranteed team-correct when at
least three mutations are correct. Two correct mutations may also suffice if
the remaining wrong answers split, but those clusters were not persisted. If
total gain memberships are `G` across 40 rows, the count of guaranteed rows is
at least:

```text
ceil(max(0, G - 80) / 2)
```

Loss bounds use the frozen loss-set overlap. Invalid outputs may weaken a wrong
cluster, so they may not be silently treated as a specific wrong answer.

## Common-Safe reachability

A path claim may be exact, bounded, or existential:

- one- and two-mutation states built only from baseline-Common-Safe, valid
  mutations are proven Common-Safe because target accuracy strictly improves,
  terminal-invalid count does not increase, and three identical baseline peers
  keep team vote unchanged;
- later edges require exact profiles unless an aggregate combinatorial proof is
  sufficient;
- existential proof does not identify a particular state and may not be used as
  a frozen API starting state.

## Outputs

The audit publishes all 24 configurations, evidence availability, baseline
identifiability, state metric bounds, proven and unresolved edge classes,
existential reachability proofs, sanitization results, and zero-API accounting.
