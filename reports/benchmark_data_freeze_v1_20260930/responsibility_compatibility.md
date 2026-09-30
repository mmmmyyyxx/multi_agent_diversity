# Responsibility compatibility

Current BBH overlapping D/N/C/V and feasibility rules retain their original
runtime identities. No algorithm, GEPA search, target selection, stopping,
memory or pattern policy was changed by this task.

All five new benchmark responsibility policies remain HOLD. The raw plurality
capabilities needed by BBH responsibility cannot be inferred from the existence
of dataset rows or a scalar evaluator. The registry continues to reject these
benchmarks before provider construction. `unified_search_ready` is now exactly
`not blockers()`, including split, provenance and output-contract gates.

`SplitAccessPolicy` and a manifest-SHA-bound `FrozenSplitReader` protect future
data composition. Existing unified `EvidenceView` already rejects any shadow
or test row in mutation, search-validation or TeamProbe evidence. Synthetic
tests exercise these guards. No new real runtime was activated.
