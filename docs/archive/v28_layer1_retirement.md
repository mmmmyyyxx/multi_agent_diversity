# V2.8 Layer1 retirement inventory and dry-run

Recovery source: `905268bad94fa9a70fb9b719c0c68d6588cd7d9d`.
Historical manifests, reports, data, checkpoints and frozen results are retained.

The planned active replacements are `search/current_layer1.py`,
`search/optimization_evidence.py`, current composition and policy wiring.
Obsolete runtime candidates for removal, after import inventory, are
`search/bounded_layer1.py`, `search/current_context.py`,
`search/selected_evidence.py`, `search/rolling_risk_memory.py` and
`search/scoped_edit_experience.py`. Do not remove shared primitives still used
by the current graph or explicitly classified historical replay.

V2.8 tests asserting six-generation chains, disjoint 3/3/6 panels, four exports,
two promotions, abstention and iterative feedback are historical conformance
tests at the recovery SHA. They are not authority for V2.9. Shared scoring,
Gradient, data, ledger, request and security checks remain current.
Inventory and classification changes must name individual modules; blanket
test deletion or allowing historical runtime execution is forbidden.

Dry-run: inspect imports, replace active graph, add V2.9 conformance coverage,
classify affected tests, then verify current composition has no retired imports.
No historical experiment execution is granted by this tombstone.

Confirmed retirement also removes DisjointGradientEvidence, FixedPeerPromotion,
CandidateEvaluationPipeline, FixedPeerCommonSafe and the old team_probe runtime
method from shared modules. Their unchanged shared measurement/value primitives
remain. Offline historical audit selectors are separate analytical code; they
cannot execute current experiments. Function-level superseded tests are listed in
v28_layer1_retired_test_cases.json and the suite inventory.
