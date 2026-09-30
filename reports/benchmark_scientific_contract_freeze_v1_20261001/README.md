# Benchmark Scientific Contract Freeze V1

BASE_SHA: `5f8c2dad0af74ad843f2dcbe533ecac30ab4f640`

`BENCHMARK_SCIENTIFIC_CONTRACT_FREEZE_V1 = YES`

`REAL_EXECUTION_READY = NO`

| Benchmark | Data | Task | System | Output | Evaluator | Aggregation | Responsibility | Unified Search |
|---|---|---|---|---|---|---|---|---|
| hotpotqa | partial/not frozen | frozen | dependency HOLD | frozen | frozen | normalized_equal_plurality_v1 | binary plurality frozen | HOLD |
| hover | not frozen | frozen | dependency HOLD | frozen | frozen | llm_evidence_union_24_v1 | HOLD | HOLD |
| ifbench | frozen | frozen | dependency HOLD | frozen | source/metric frozen; runtime HOLD | llm_equal_raw_responses_v1 | HOLD | HOLD |
| math | not frozen | frozen | pinned offline scorer | frozen | frozen | equal_equivalence_plurality_consistency_v1 | binary plurality frozen | HOLD |
| pupa | not frozen | frozen | dependency HOLD | frozen | fake oracle frozen; real judge HOLD | HOLD | HOLD | HOLD |

Task protocols are independent of data and system readiness. IFBench source/control-flow readiness does not imply availability of its language resources. PUPA has a member pipeline/scorer contract; team semantics remain HOLD.

BBH parity: 1,024 synthetic five-member vote states and 32 feasibility masks; D/N/C/V, lane, raw legal portfolios, failure-discounted scores and chosen member agree with frozen V4. Existing BBH fake replay also checks exact evidence, GEPA and transition behavior. Five synthetic benchmark replays passed; these are contract checks, not efficacy observations.

Install `requirements-benchmark-evaluators.txt` before clean-checkout tests. Tests need no ignored raw datasets. Verification used the credential-free, pre-import network guard. Dataset downloads, real materializations, provider/judge, Validation/Test model calls and formal executions were all zero. Software dependency installation, git fetch and two public source inspections used network access; their exact total HTTP request count was not captured.

The HoVer normalization source is [DSPy 2.6.27](https://raw.githubusercontent.com/stanfordnlp/dspy/2.6.27/dspy/dsp/utils/metrics.py). Other benchmark program/metric source objects were inspected from the previously retained `gepa-ai/gepa-artifact@cbefbc1aa0f43dd39874ec4bf42211365dbda42e`; none of its provider/retrieval modules were imported.

See `preflight_audit.json` and `remaining_scientific_blockers.md` for exact holds. Existing data identities and historical reports are preserved. Publication is local-commit-only; this task does not authorize a push.
