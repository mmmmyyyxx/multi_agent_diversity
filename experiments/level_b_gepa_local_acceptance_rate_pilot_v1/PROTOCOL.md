# Level-B GEPA local acceptance-rate pilot v1

This is an independent prospective Layer-1 mechanism pilot, not a continuation
of the completed real-provider canary. It asks how often the current official
Level-B GEPA search produces a strict local improvement for the current
responsibility-conditioned task.

The pilot freezes Seed79 and the anti-overfitting fold-A-plus-B Optimize100
split. One P0 team is initialized, then the update-zero scheduler selects its
top three distinct actionable members. Those three member/context pairs are the
three local parent-tasks. They all retain the same immutable P0 team; no prompt
is written back between parents.

Each parent receives exactly four actual proposal attempts. Stopping uses the
pinned GEPA public stop-callback interface and the authoritative proposal-end
callback count, so skipped perfect minibatches do not silently shrink the
denominator. Each parent also has a frozen 120 local metric-call ceiling. If the
three-parent or four-proposal denominator is not realized, the experiment is
HOLD and is not retried.

The main outcomes are proposal-level minibatch `delta_local`, official strict
improvement acceptance rate, and the fraction of parents with at least one
accepted mutation. Diagnostic outcomes are newly-fixed, newly-broken,
preservation loss, token-set edit similarity, and allowlisted reflection-pattern
concentration. The before/after effects use the official three-example
reflection minibatch; they are not full-team or held-out effects.

The official GEPA v0.1.1 search core, reflection template, candidate component,
parent selection, Pareto state, batch sampler, strict acceptance, Solver model,
and reflection model are unchanged. The experiment ends at Layer 1. It invokes
no TeamMiniBatch, Full team evaluation, Shadow50, Validation50, Test50, or team
commit. It has one fresh attempt, no resume, and no experiment-level retry.

The frozen result labels are:

```text
accepted_mutations > 0 -> LOCAL_STRICT_IMPROVEMENT_OBSERVED
accepted_mutations = 0 -> NO_LOCAL_STRICT_IMPROVEMENT_OBSERVED_IN_PILOT
```

Neither label alone establishes that GEPA is generally effective or ineffective.
