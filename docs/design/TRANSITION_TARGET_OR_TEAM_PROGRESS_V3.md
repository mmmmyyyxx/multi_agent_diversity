# Initial competence target-or-team transition V3

The sole active research architecture is Unified Team Prompt Search.
`initial_competence_target_or_team_progress_v3` is retained unchanged in V2.4.
Historical V2.2 execution requires its original source; current admission uses
the complete structured prompt policy bundle.

For target member t, initial competence I, incumbent competence P, Full
candidate competence C and Full team Vote V, deployment eligibility is:

```text
C[t] >= I[t]
AND V_candidate >= V_parent
AND (C[t] > P[t] OR V_candidate > V_parent)
AND existing versioned invalid-output / terminal-invalid safety guard
```

The floor is the actual initialization count on the complete frozen Optimize
membership, independent of aggregation. Its support, metric, evaluator and state
identity survive every commit and rollback and cannot rebase. Missing or
nonfinite measurements fail closed. The existing prediction validity policy is
unchanged: guard-enforcing modes require terminal-invalid delta <= 0; the frozen
MATH recovery policy scores terminal-invalid as incorrect/no vote and keeps its
invalidity telemetry observation-only. No parser, retry or invalidity weighting
is amended here.

TEAM progress allows incumbent target decline above its immutable floor when
Vote strictly increases. TARGET progress requires strict target improvement and
non-regressing Vote. Both-positive is TARGET_AND_TEAM. Neutral, team regression,
below-floor or a violation of the bound safety policy cannot deploy.

Deployment ranking uses Full team Vote, then Full target competence, then
existing newly-broken, soft-vote and invalidity metrics and deterministic prompt
hash. No Layer1/GEPA local score enters this key. Promotion remains the existing
TeamProbe rule: any positive responsibility, target, Vote, broad or net-team
signal may promote subject to its catastrophe guards, with at most two Fulls.
Eligibility is reachability, not a guarantee that every positive candidate
wins the two-slot budget. Full diagnostics and final selection call the same
versioned transition policy; there is no second strict-Vote gate.

Only the selected Full winner enters private Shadow. Shadow retains Vote delta
>= 0 and target loss >= -2; target-only Full progress may commit with zero or
bounded negative Shadow target gain. Requiring target gain replication on
Shadow is a separate scientific decision. No held-out sample or target metric
feeds search, Memory or allocation.

Completed atomic commit is Memory success, including TARGET progress with
team delta zero. Transaction preparation/validation precedes commit and memory
application follows successful deployment. A selected but rejected or rolled
back candidate is not a success. Allocation telemetry retains realized team
gain and adds realized target gain and TEAM / TARGET / TARGET_AND_TEAM / NONE
progress paths; these fields have no scheduling or stopping read point.

Scientific stopping remains `team_epoch_no_commit_v1`: any completed commit
resets no-commit patience, independently of which progress path passed.
Operational opportunity/provider/transport ceilings remain distinct from the
scientific stop. The strict-Vote N-commit proof cannot establish saturation
under target-or-team progress. Current preparation freezes a finite opportunity
horizon with coupled token admission; resource truncation remains incomplete.
V2.4 uses its fresh structured binding and exact single-use authorization, with
no historical source, cache, competence or authorization fallback. Validation
and Test remain independently locked. Synthetic fixtures establish conformance.

WHO, raw overlapping D/N/C, V=max(4D,2N,C), failure discount, Pattern selection,
Gradient generation, Layer1 budgets, five equal-weight plurality members,
fixed peers, two promotions, winner-only Shadow and atomic one-member commit
remain unchanged. Efficacy and generalization require future governed evidence.
