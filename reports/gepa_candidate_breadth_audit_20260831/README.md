# GEPA-style Candidate Selection and Proposal Breadth Audit

## Scope

This report combines two fixed-evidence stages on the two V18 harmful
Common-Safe feasible-set-gap parents: Seed59 update 3 and Seed61 update 5. It
changes no method, target selector, Common-Safe rule, ranking, M20/M2F
mechanism, parent, target, peers, trajectory, validation policy, or test policy.

## Phase A: retrospective selector audit

The train-only GEPA-style selector retained the current Common-Safe Top-2 and
then applied per-residual best-in-class `(DeltaVote, DeltaMargin,
target-correct)` coverage. Selection was frozen before existing validation
labels were read.

```text
historical parents = 2
historical feasible candidates = 7
Top-2 candidates audited = 4
winner changes = 0/2
validation Vote improvement = 0
validation Oracle improvement = 0
diagnosis = CANDIDATE_SELECTION_NOT_PRIMARY
```

The historical winner was already the residual-frontier choice in both harmful
pools. This supports a feasible-set/candidate-generation quality gap rather
than a ranking mistake for these witnesses.

## Phase B: prospective N=2 versus N=4 breadth pilot

Each frozen parent requested one four-source proposal. N=2 was preregistered as
the nested first-two-source pool plus one unchanged loss-blind revision per
valid source; N=4 included all requested sources and their revisions. Train
decisions had to freeze before validation. No prompt could be committed.

The canonical gate passed with zero blockers, zero state mutation, zero prompt
commits, zero test calls, and unchanged source identity. However, both parents
terminated before Student generation:

```text
requested sources = 8 total
actual valid sources = 0
Seed59 update3 = critic_semantic_rejection_exhausted
Seed61 update5 = critic_semantic_rejection_exhausted
Teacher calls = 4
Critic calls = 4
Student calls = 0
Solver calls = 0
```

Consequently neither N=2 nor N=4 produced an evaluable candidate pool. There
were no feasible, zero-loss, or lower-loss candidates and no frozen winner to
evaluate on validation.

## Frozen classification

```text
final label = NO_PROPOSAL_BREADTH_SIGNAL
candidate breadth effect interpretable = false
interpretation = not_evaluated_downstream_mutation_due_critic_semantic_exhaustion
```

This label is not evidence that N=4 is ineffective. It means the proposed
breadth intervention never reached mutation generation under the unchanged
Teacher/Critic pipeline. The experiment therefore localizes the immediate
bottleneck further upstream: simply requesting more mutations cannot improve
the feasible set when the unchanged semantic gate releases no plan.

The initial execution produced the same zero-source shape but did not persist
the terminal funnel. It is retained as an invalidated representation attempt
and excluded from scientific counts. The runner was changed only to persist
the funnel, then refrozen and rerun in a fresh root. Authoritative cost is 8
role calls; total incurred cost including the invalidated attempt is 16 role
calls. Neither attempt made Solver, validation, or test calls.

## Answer

For the tested harmful pools, existing-candidate selection is not the primary
problem. Candidate-generation quality remains the supported bottleneck. The
specific N=2 versus N=4 mutation-breadth effect remains unresolved because the
unchanged Critic rejected both plans before any candidate was generated.
