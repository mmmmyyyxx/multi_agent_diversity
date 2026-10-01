# V2 Pattern/Memory factorial V1 — preexecution STOP

```text
BASE_SHA = 0da8bf96e957ae227fdbf988ad6e3d67cd190ef7
PREEXECUTION_SHA = NOT_CREATED
EXECUTION_SHA = NOT_CREATED
RESULT_REPORT_SHA = enclosing publication commit (see Git history)
STATUS = PREEXECUTION_BLOCKED
REAL_API_USED = NO
DATASET_DOWNLOADS = 0
TEST_UNLOCKED = NO
```

The task requires A0/A1 to share the search metric budget. Current V1 real
composition requires enabled GEPA saturation and calls `optimize_saturation`.
That wrapper passes `max_metric_calls=None`, despite a declared task budget of
36. V2 preserves its complete-epoch observer but additionally forces the public
GEPA metric limit to 36. The zero-API public-call capture confirms this difference
before any Solver, reflection or Pattern call.

The A0/A1 budget-equivalence hard gate therefore fails. Per the task's STOP rule,
no canary, baseline, mechanism arm, Validation or Test is run, no execution source
is frozen, and no authorization is consumed. No source, prompt, dataset, method
identity or historical artifact is amended to bypass this failure. This is a
preexecution protocol conflict, not a negative efficacy result.

Matching A0 to the 36-metric bounded execution would require an explicit reference
contract/identity decision. Retaining its historical saturation execution would
require an explicit amendment to the matched-budget preregistration. Neither
decision is inferred from observed results; no new experimental efficacy exists.

- [Budget invocation proof](preexecution/budget_equivalence_gate.json)
- [Data/model prerequisite identities](preexecution/prerequisite_identity_audit.json)
- [All 65 requested fields](preexecution/final_requested_fields.json)

No implementation or preexecution freeze commit was created. Publication was
explicitly requested after review. This report does not claim
PREEXECUTION_FROZEN readiness; the inherited scientific source is unchanged.
The four preexecution JSON files retain the original audit snapshot, including
its pre-publication Git status and zero push count. Only this publication note,
the report hash manifest, and repository audit/index metadata are updated for
publication. The unrelated retry2 directory remains untracked.
