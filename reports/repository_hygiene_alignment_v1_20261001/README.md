# Repository Hygiene & Governance Alignment V1

BASE_SHA: `ec993b13a252699f19b93348b01570ffd9bea879`

Local commit only. No push, dataset download, model call or experiment execution.

| Area | Before | After | Status |
|---|---|---|---|
| AGENTS.md | Mixed current/history, 782 lines | Stable governance, 190 lines | PASS |
| CURRENT_SPEC | Mixed Unified/v15/two-layer | Active Unified and benchmark interface invariants | PASS |
| method.md | Historical v15 with stale active wording | Current Unified conceptual overview | PASS |
| historical specs | Mixed into current authority | 9 verbatim archives, 40 historical invariant IDs indexed | PASS |
| registry | 27 nodes, stale lifecycle labels | 179 v2 nodes, explicit unknown evidence and eras/kinds | PASS |
| lineage | 30 legacy edges | 179 nodes, 58 typed edges; regenerated view | PASS |
| manifests | Heterogeneous frozen formats | 54 preserved, 39 shared-v1 / 15 bespoke; new v2 schema/template | PASS |
| reports | No complete index | 185 indexed paths, historical files unchanged | PASS |
| scripts | Mixed operational/historical | Classified; 90 utilities marked historical, original paths kept | PASS |
| tests | Current and private replay mixed | Explicit current/full-historical suites and markers | PASS |

## Verification

392 distinct current cases passed; 4 skipped because their real-data manifests are not frozen.
The clean full current run passed 389 cases; the final 98-case governance retest also covers three new source-scope guards. Counts do not double-count retests.
1,468 historical cases were explicitly deselected (1,404 replay and 64 cases in eight private-artifact modules). Full historical collection succeeded with 1,864 cases; full historical execution was not performed or claimed to pass.
Compileall, manifest/invariant/registry/lineage/archive/generated-view checks and offline import guards passed.
The credential-free guard was active before application import, with zero network attempts in normal guarded work.

## Preservation and limits

No tracked file was deleted or physically moved. All historical reports, manifests, datasets and membership files retain their original bytes and paths. Archived bodies preserve baseline Git text with source provenance. Historical utilities received comments only; their Python AST or PowerShell body is unchanged. All 113 runtime assignments retain identical names and values.
The four stale labels were resolved by cited lifecycle evidence; the old scientific classifiers were preserved. Unknown report-container metadata remains STATUS_UNRESOLVED. Registry inventory identities never imply a new execution or result.
Current source identity separates scientific source, governance plus operational bootstrap, benchmark contracts and explicitly bound dataset manifests. Package bootstrap still imports historical compatibility code; it remains covered by governance receipts, while historical controllers, reports and archives do not enter the current scientific hash. New real execution still needs a reviewed dynamic dependency closure, frozen governed prep and separate authorization.
Historical v5 identity remains available through its existing function and explicit `--family historical-v5`; exact frozen replay uses the original source SHA. Five migrated benchmarks remain HOLD and REAL_EXECUTION_READY remains NO.
The unrelated untracked retry2 directory was neither changed nor staged. The 89 duplicate groups and unknown files were retained.

```text
REPOSITORY_CURRENT_SURFACE_ALIGNED = YES
EXPERIMENT_REGISTRY_ALIGNED = YES
EXPERIMENT_LINEAGE_ALIGNED = YES
HISTORICAL_EVIDENCE_PRESERVED = YES
CURRENT_SOURCE_IDENTITY_CLEAN = YES
SCIENTIFIC_BEHAVIOR_CHANGED = NO
DATASET_DOWNLOADS = 0
REAL_LLM_CALLS = 0
VALIDATION_CALLS = 0
TEST_CALLS = 0
REAL_EXECUTION_READY = NO
```
