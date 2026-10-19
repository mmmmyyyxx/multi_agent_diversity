# Failure-registry evidence categories V1

Scope: `docs/failures/registry.yaml`, validated by
`multi_dataset_diverse_rl.governance.registries.validate_failure_registry`.
These are evidence categories, not an ordered scale of scientific strength.
Lifecycle/status and evidence category are separate fields. `MITIGATED` does not
waive metadata validation, prove efficacy, or grant execution authorization.

| Category | Meaning | Does not establish |
|---|---|---|
| `hypothesized` | A proposed explanation awaiting adequate verification. | Occurrence, verified implementation behavior or causality. |
| `observed` | Behavior or a failure was observed in the stated artifacts/runtime context. | That the proposed cause is correct or that a mitigation improves performance. |
| `source_verified` | The stated implementation/flow property was checked against traceable source and linked inspection or offline regression evidence. | A real API failure, historical execution incidence, causal attribution, model efficacy, or generalization. |
| `causally_supported` | The stated cause has separate supporting intervention/comparison evidence, with its scope and limitations in the cited evidence. | Universal causality or an effect outside the studied conditions. |

For `source_verified`, supply nonempty `evidence` and `regression_tests` reference
lists. Source locations, revision/context and what was actually checked belong
in the cited evidence. Existing references must exist; required fields, status,
duplicate-ID and reference validation remain enforced. Unknown labels fail
closed. The validator checks metadata conformance, not whether prose is true or
an intervention identifies a cause. Human scientific review remains necessary.

The two existing ablation entries concerning the diagnostic universe and formal
adapter conformance use this category: their cited V5/V6 documents and regression
tests establish implementation/flow findings. Their original records, mitigation
status, evidence labels, historical run artifacts and scientific conclusions are
preserved. Registering this already-used label is compatibility support, not a
retroactive conversion to `observed` or `causally_supported`.

No model, seed, split, cache, retry, scoring, candidate Guard, scheduling,
acceptance, stopping or resource rule changes. No API or Test access is granted.
Fresh executable preparation after this governance change must bind a new source
SHA and hashes; old frozen preparations and authorizations are never rewritten.
