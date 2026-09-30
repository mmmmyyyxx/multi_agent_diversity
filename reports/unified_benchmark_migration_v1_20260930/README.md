# Unified benchmark migration V1 — zero API

Base: `b8555b9005fb117a086c639e51e0a5e6f9ba1443`. This is a software and scientific-capability audit, not a new experiment. Existing BBH search and historical Formal V3 were not changed.

| Benchmark | Dataset adapter | Evaluator | Aggregation | Responsibility | Unified search |
| --- | --- | --- | --- | --- | --- |
| HotpotQA answer-only component | YES, hash-checked JSONL + input adapter | YES, answer EM/F1 component | candidate normalized-answer plurality; policy unfrozen | NO | NO |
| HoVer | NO, provenance blocked | NO | NO | NO | NO |
| IFBench | NO, provenance blocked | NO | NO | NO | NO |
| PUPA | NO, provenance blocked | NO | NO | NO | NO |
| MATH | NO, provenance and trusted equivalence blocked | NO | NO | NO | NO |

These flags concern the requested project benchmark protocol, not the mere existence of an upstream benchmark. HotpotQA's adapter implements only answer scoring; it does not assert that answer-only is the project's selected task. The other four have explicit adapters that raise `BENCHMARK_PROVENANCE_NOT_FROZEN` before parsing or scoring. All five preflight as `HOLD_PRE_PROVIDER`; they cannot enter Unified Search.

`BENCHMARK_INFRASTRUCTURE_READY = YES` for registry, hash-checked local materialization, split auditing, cache/request identity and pre-provider capability gates. `HOTPOTQA_UNIFIED_SEARCH_READY = NO`; `HOVER_UNIFIED_SEARCH_READY = NO`; `IFBENCH_UNIFIED_SEARCH_READY = NO`; `PUPA_UNIFIED_SEARCH_READY = NO`; `MATH_UNIFIED_SEARCH_READY = NO`. `REAL_EXECUTION_READY = NO / HOLD_PRE_PROVIDER`.

`SCIENTIFIC_DECISION_REQUIRED`: select exact upstream versions/configs and project tasks; obtain or create authorized local materializations and frozen split assignments; bind official/fidelity-tested evaluators for structured/constraint/math/privacy tasks; freeze aggregation and responsibility semantics for each non-BBH task. No new split, D/N/C generalization, or real API run was authorized here.

The BBH fake dynamic replay now covers two commits, successor parents/history and terminal saturation. Test evidence and exact hashes are in this directory. The unrelated untracked retry report was left untouched. No push was requested or performed.

Guarded focused tests: `30 passed`. Guarded broad suite, excluding only eight documented historical private-artifact files: `1661 passed, 4 skipped`. Unfiltered run: `1686 passed, 15 failed, 13 errors, 4 skipped`; every failure/error arose from unavailable private frozen inputs. Network attempts and real API calls remained zero.
