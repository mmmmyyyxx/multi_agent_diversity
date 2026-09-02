# Current Implementation Specification

This is the normative implementation specification for the current algorithm.
It is not a paper narrative or an experiment report. Runtime identifiers and
frozen constants remain authoritative in
`multi_dataset_diverse_rl/versions.py`.

## Canonical Runtime

The canonical runtime is `member_aware_peer_state_v15`, checkpoint v25:
Repairability-Adjusted Dual-Target Prompt-Team Optimization. It retains the
semantic hard-veto Teacher-Critic-Student pipeline. Experimental evidence does
not modify this section without a separate method change and version update.

### Identity and team semantics

- **INV-ID-001** — Runtime protocol identifiers and frozen weights MUST be read
  from `versions.py`; no parallel constants authority is permitted.
- **INV-VOTE-001** — The team has exactly five equally weighted members.
  Plurality ties abstain and are incorrect. For each row, `G` is valid gold
  support, `H` the largest valid wrong cluster, and the vote is correct iff
  `G - H > 0`.

### Dataset lifecycle and isolation

- **INV-DATA-001** — Training diagnosis and candidate rollout use only the
  frozen optimization split. The canonical runtime performs no validation
  rollout or validation-based checkpoint selection; the final active state is
  selected automatically.
- **INV-TEST-001** — Test data MUST NOT influence search, candidate generation,
  candidate selection, arm selection, validation selection, or final-state
  selection. If enabled, test is evaluated at most once after training freeze.

### Responsibility and target scheduling

- **INV-RESP-001** — Only vote-wrong rows create residuals. Only currently
  wrong members are eligible. Eligibility is the lexicographic argmax of
  counterfactual `(DeltaV, DeltaM)` with exact ties retained; gain, wait,
  history, memory, and load cannot alter eligibility.
- **INV-ROUTE-001** — Each serviceable residual is routed to exactly one legally
  eligible member. Service portfolios are disjoint. Each member exposes one
  active lane and only its routed same-lane slice.
- **INV-TARGET-001** — S1/S2 select the two highest-ranked distinct actionable
  members under the repairability-adjusted W1 score. One actionable member
  degrades to one branch; none causes `no_actionable_responsibility`.
- **INV-REPAIR-001** — Normal branches update attempt/feasible/failure state.
  Operational failures do not. Only an accepted changed team state resets all
  state-local repairability counters.

### Branch construction and TCS

- **INV-BRANCH-001** — S1/S2 construct two independent branches from the same
  parent team, peers, responsibility snapshot, routing, lanes, and profiles.
  Each branch has two candidates; branches cannot observe each other's changes.
- **INV-TCS-001** — Program selects numerical evidence; Teacher proposes one
  bounded repair hypothesis; Critic applies canonical semantic blockers;
  Student realizes replacement prompts; rollout supplies empirical value.
  Student sees no raw peer output, identity, score, wait, load, or empirical
  rejection history. Invalid recovery is bounded.

### Candidate evaluation, Common-Safe, and ranking

- **INV-COMMONSAFE-001** — With four peers fixed, a candidate must preserve
  target correct count, preserve team vote correct count, strictly improve at
  least one of them, and not increase terminal-invalid count.
- **INV-RANK-001** — Branch candidates and branch winners use the versioned
  Common-Safe ranking. Cross-branch comparison cannot use target absolute
  accuracy, raw lane utility, or raw portfolio size.
- **INV-COMMIT-001** — At most one prompt commits per update. The global winner
  is chosen before atomic prompt/profile/anchor update, successor diagnosis,
  repairability reset, responsibility refresh, persistence, and audit. Any
  failure rolls all affected state back.

### Artifacts and access evidence

- **INV-ARTIFACT-001** — Publishable artifacts may contain hashes, identifiers,
  counters, categories, metrics, and token totals, but not prompts, questions,
  gold/model answers, raw responses, secrets, endpoints, caches, checkpoints,
  or absolute host paths.
- **INV-AUTH-001** — API use fails closed unless the user explicitly authorizes
  it and the frozen manifest independently authorizes the phase and role with a
  matching preregistration hash and frozen budget.
- **INV-MANIFEST-001** — Experiment design, API authorization, budget,
  validation/test policy, selection rule, source identity, evidence type, and
  lifecycle MUST be explicit. Protocol changes after freeze require an
  amendment; historical artifacts are never rewritten.

## Reduced matrix

The canonical matrix is Static, S0 Generic, S1 member-aware dual-target, and S2
responsibility-conditioned dual-target. S2 is the full method. Common-Safe is a
shared write-back policy rather than a separate module. Legacy and auxiliary
settings require their explicit opt-ins.

## Next Candidate Architecture

The latest fixed-parent four-arm experiment selected this candidate pipeline:

```text
Teacher-Clean
-> deterministic hard gate
-> Student
-> empirical rollout
```

Its status is exactly `SELECTED_FOR_NEXT_ONLINE_VALIDATION`. Evidence supports
higher Student throughput, more feasible candidates, and better target transfer
within the frozen local experiment. It does **not** demonstrate validation Vote
improvement, online trajectory superiority, test improvement, or promotion to
canonical runtime. The canonical Critic path above remains unchanged.

## Authority map

Machine-readable mirrors of these invariant IDs live in
`docs/design/invariants.yaml`. Experiment-specific deltas belong in manifests;
evidence and conclusions belong in reports. `method.md` is explanatory prose.
