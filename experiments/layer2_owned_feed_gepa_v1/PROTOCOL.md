# GEPA search core with Layer-2-owned evidence V1

This prepared canary compares `NATIVE_GEPA_CONTROL` with
`GEPA_SEARCH_CORE_WITH_LAYER2_EVIDENCE_V1`. The treatment replaces GEPA's
optimizer-local example selection with one immutable Layer-2 responsibility
packet while retaining the pinned official GEPA v0.1.1 population, Pareto,
parent-selection, reflection, proposal, local acceptance and lineage logic.

Layer 2 freezes the target member, responsibility, repair evidence,
preservation evidence, local-evaluation evidence and ordered batch schedule
before Layer 1 begins. GEPA may fetch only those IDs and may not reorder,
expand, replace or fall back to native sampling. TeamMiniBatch, Full,
Common-Safe, Shadow and write-back are unchanged downstream stages.

The future small real-provider canary asks only whether Layer-2 evidence can
drive real GEPA reflection/mutation through the official search core. It must
record packet hash, exact delivered batch IDs, proposal reach, candidate Solver
reach, local delta and acceptance/rejection. Validation50 and Test50 are
inaccessible. This preparation authorizes no provider call.
