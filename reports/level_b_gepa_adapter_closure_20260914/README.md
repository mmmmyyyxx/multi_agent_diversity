# Level-B GEPA adapter closure

Gate: **PASS**. The official GEPA source remains unchanged at v0.1.1 and the
backend is `LEVEL_B_API_COMPATIBLE_ADAPTATION`, not a Level-C fork.

GEPA now optimizes exactly one API component: `decision_procedure`. The existing
COMMON_SOLVER_CONTRACT_V1 evaluator instantiates the immutable solver shell and
output interface exactly once. The supported seams adapted here are the adapter,
local train/validation domain, evaluator, reflection template, provider, budget,
callbacks, cache, and seed.

Official GEPA still owns population, Pareto/frontier maintenance, parent
selection, reflection mutation and minibatches, local acceptance, lineage, and
merge semantics. Layer 2 controls none of those decisions; GEPA knows neither
member selection nor persistent realizability.

The deterministic fake-provider path proves a valid changed procedure reaches
Solver evaluation, is accepted by official GEPA, survives the aligned strict
4/4/4 TeamMiniBatch12, reaches full evaluation and Shadow, and commits exactly
once. Four invalid classes fail before Solver rollout. Candidate validation and
deduplication occur before Top-K.

The canonical `member_aware_peer_state_v15` runtime is unchanged. Real API, Validation,
and Test calls are all zero. A fresh canary is technically safe to authorize,
but remains forbidden until separately authorized.

Local optimizer contract hash: `ae991a97dfefd73eb32521ddfbb53db0624eea16f58e2a59ad4b063d5325600a`.
