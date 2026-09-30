# Formal V3 Attempt4 execution scheduling

Formal V3's scientific method and Stage 0 split policy remain frozen. Attempt4
records an execution implementation change from source
`9e498496abff9228bd10697d8d97d36f56fca3f9` to
`6b16c8abe8717895c26a6a05bbdbc2e9e636388e`.

The old GEPA adapter evaluated each local batch row synchronously. Its local
effective Solver concurrency was one even though the Formal runner configured
eight. The new adapter passes one complete GEPA batch to
`SystemLocalSolverEvaluator.evaluate_batch`. That evaluator validates every
row against the frozen Optimize probe, sets one batch-wide ledger stage, and
uses the existing `FixedProbeEvaluator.evaluate_prompt_indices` gather path.
The existing Solver semaphore caps physical requests at the manifest's
execution-only value of 16. Results are returned to GEPA in original batch
order. The run-local prompt-question cache, exact-request cache, Solver request
identity, and `evaluation_replica_seed` cache identity remain in place.

GEPA reflection, proposals, candidate acceptance, Pareto updates, Layer2
opportunities, candidate competition, Shadow checks, and atomic commits retain
their sequential dependency order. `cache_evaluation=False` and cross-seed
cache isolation remain unchanged. The pinned official GEPA package and Layer2
scientific rules are unchanged. This scheduling change is an execution detail,
not a method contribution.

The [pre-execution audit](../../reports/formal_v3_attempt4_concurrent_execution_preexecution_20260930/README.md)
contains the source closure, deterministic old/new semantic comparison, fake
latency calibration, cache and ledger checks, and the six execution-gated prep
identities. No real Attempt4 cell is authorized or executed by this preparation.
Validation50 and Test50 remain sealed.
