# Machine-readable capability interpretation

`multi_dataset_diverse_rl.benchmarks.registry.BENCHMARKS` is the executable source for the table. `BenchmarkSpec.blockers()` is checked before the orchestrator touches state or search providers. Existing BBH stays on its current composition and is not inferred from this new registry.

| Benchmark | Loader | Evaluator | Frozen split | Plurality representation | LLM aggregate | Current D/N/C | Unified Search |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BBH current path | YES | YES | YES in frozen historical manifests | YES | optional infrastructure | YES | YES for existing fake/current composition |
| HotpotQA answer-only component | hash-checked JSONL generic loader | answer EM/F1 only | NO | normalized text in fixture | infrastructure only | NO | NO |
| HoVer | blocked | blocked | NO | NO | unselected | NO | NO |
| IFBench | blocked | blocked | NO | NO | unselected | NO | NO |
| PUPA | blocked | blocked | NO | NO | unselected | NO | NO |
| MATH | blocked | blocked | NO | NO without equivalence classes | unselected | NO | NO |

`BENCHMARK_IMPLEMENTED` is therefore false for all five as complete project protocols; HotpotQA has a separately tested answer-only evaluation component. `AGGREGATION_READY`, `RESPONSIBILITY_READY`, and `FULL_UNIFIED_SEARCH_READY` are false for all five. This does not classify upstream benchmark quality; it records the missing project freeze.
