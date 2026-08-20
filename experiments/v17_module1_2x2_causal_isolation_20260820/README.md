# Phase A result

The V17 Module-1 2x2 fixed-parent causal isolation is execution-ready and has
made zero API, validation, and test calls.

Frozen inventory:

- 6 pairwise-distinct V17 S2 parent-team states;
- 2 concentration witnesses, 2 throughput witnesses, 2 neutral controls;
- 4 cells and 24 parent-cell evaluations;
- 48 independent target branches;
- 96 maximum source candidates (2 targets x 2 sources per cell);
- exactly one loss-blind generic revision for every valid source;
- immutable fixed-parent WOULD_COMMIT simulation with Common-Safe and max-one;
- realized validation vote and oracle deltas plus a five-label classifier;
- no candidate test evaluation.

The zero-API preflight reconstructs all six prompt teams and 75-row profile
states, independently replays the W1 total order and dual round-robin targets,
and constructs all 48 proposal contexts. A/B are pure
`AccuracyDiagnosisContext`; C/D are `PeerStateDiagnosisContext`. The
SingleLane/M20 context path is not used.

Phase B is intentionally authorization-latched. After the source-freeze commit,
`freeze_v17_module1_2x2_phase_a.py` creates a private project-local registry,
preflight record, and source manifest under ignored `runs/`. The PowerShell
wrapper checks that freeze before allowing the low-API run.

Verification at Phase-A handoff:

```text
new focused tests: PASS
canonical tests: 616 passed, 3 known historical-artifact failures
new failures: 0
API/model calls: 0
validation calls: 0
test calls: 0
```

The three existing failures remain the known missing historical cache coverage
test and two tests that require an API key while initializing an old online
compatibility-repair checkpoint. They are unrelated to this probe.
