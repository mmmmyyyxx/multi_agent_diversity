# Level-B GEPA real-provider canary

Gate: **PASS**  
Classifier: **PROPOSAL_CONTRACT_STILL_BLOCKS_EMPIRICAL_SEARCH**

This single-parent engineering canary tests empirical path activation, not scheduler or accuracy efficacy. Validation50 and Test50 calls are zero.

The launch transaction, lifecycle, accounting ledger, and official frozen audit
all passed. The real proposer made three attempts, but no proposal reached the
candidate Solver. The frozen summary records all three as unmaterialized because
pinned GEPA does not retain rejected proposals in `result.candidates`.

A zero-API, read-only reconstruction from the private run log classified all
three rejected proposals as `output_contract_contamination`; two also triggered
the `append_only` failed check. Only proposal hashes, character counts, and
sanitized classifications are published in `posthoc_rejection_audit.json`.
This supports a proposer contract-compliance bottleneck for this canary, not a
claim about downstream candidate quality or method efficacy. It also identifies
a remaining observability gap: rejection categories must be recorded before
pinned GEPA discards rejected proposals.
