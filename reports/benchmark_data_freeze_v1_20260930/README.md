# Benchmark Data Freeze V1 — partial completion

Base: `789792383db65eb886806700fc12dbd73d5bbb5d`.

The user deferred remaining downloads after public-network failures. IFBench
is actually frozen; the other four benchmarks have download recipes and
synthetic split tests, with no claimed real dataset freeze. This report is a
snapshot of that state. No real LLM/search/judge/evaluation was executed. This
task creates a local commit only.

| Benchmark | Source frozen | Downloaded | Split frozen | Exactness | Evaluator | Aggregation | Responsibility | Unified Search |
|---|---|---|---|---|---|---|---|---|
| HotpotQA | Revision resolved; bytes pending | One of two train shards | NO | Pending; planned SOURCE_LOGIC_REPRODUCED | Answer EM/F1 component; system HOLD | HOLD | HOLD | NO |
| HoVer | Revisions resolved; bytes pending | NO | NO | Pending; planned SOURCE_LOGIC_REPRODUCED | Retrieval metric provenance audited; integration HOLD | HOLD | HOLD | NO |
| IFBench | YES | YES | YES | EXACT_SOURCE_AND_INSTANCE_ALIGNED | Fractional constraint metric provenance audited; integration HOLD | HOLD | HOLD | NO |
| PUPA | Revision resolved; bytes pending | NO | NO | Pending; planned SOURCE_LOGIC_REPRODUCED | Fake judge contract PASS; real scorer HOLD | HOLD | HOLD | NO |
| MATH | Mirror revision resolved; bytes pending | NO | NO | Pending; PROJECT_PROPOSED | UNFROZEN | HOLD | HOLD | NO |

## Actual counts

| Benchmark | Source rows observed | Search | Shadow | Validation | Test |
|---|---:|---:|---:|---:|---:|
| HotpotQA | Complete train count unverified | — | — | — | — |
| HoVer 3-hop | Unverified | — | — | — | — |
| IFBench | 14,971 train + 294 test | 150 | 300 | 0 | 294 |
| PUPA | Unverified | — | — | — | — |
| MATH | Unverified | — | — | — | — |

The remaining **planned**, unobserved split counts are 150/300/0/300 for
HotpotQA, HoVer and MATH, and 111/111/0/221 for PUPA. The planned canonical MATH
source counts are 7,500 train and 5,000 test; its source loader fails closed
unless those counts are actually observed. No distribution-based resampling.

## IFBench identity

Pinned artifact: `gepa-ai/gepa-artifact@cbefbc1aa0f43dd39874ec4bf42211365dbda42e`.

- Original train: 17,115,408 bytes; SHA256 `a5ec13223a93879b7172da783d54669d5873dc4632b57fd6d961730c9679fc8c`.
- Original test: 419,877 bytes; SHA256 `11c3d683dcc7f4908a4d3cacd05c9a8bbd5484af2f8fde969e7abe2b8bad3e34`.
- Manifest SHA256: `7076ebb0dac504fd279093043dd3851be0d2d577af8d02a530b08a7840181e2d`.
- Ordered IDs, group hashes, canonical source hashes and materialized sizes:
  [ifbench.json](../../data/benchmark_suite_v1/manifests/ifbench.json).

The independently downloaded bytes' Git blob IDs match the pinned repository
objects. Source train contains more than the first 600 entries used by this
protocol; the whole source is retained locally, without expanding search or
shadow membership. OOD instruction categories are disjoint (25 source train
categories; 57 source test categories). No test prompt is included in reports.

Cross-split selected ID/content/input-hash audit is DISJOINT. Upstream
duplication counts are reported separately, without silently removing rows.
Raw examples and materialized data remain ignored.

## Remaining work and readiness

[Download instructions](../../data/benchmark_suite_v1/README.md) and
[pinned pending plan](../../data/benchmark_suite_v1/manifests/download_plan.json)
allow the user to finish public-data materialization later. The complete
`suite.json` is deliberately absent until all five actual freezes pass.
Existing IFBench artifacts cannot be overwritten with a different freeze.

MATH remains `AGENTGRAD_MATH_EXACT_SPLIT_UNRESOLVED`; the proposed fallback
does not claim AgentGrad/MACM sample identity. The preferred official archive
returned HTTP 401; the qwedsacf repository did not expose original train/test
membership. The pending canonical mirror is EleutherAI's seven subject
configurations, with an explicitly recorded subject/row ordering.

Only IFBench is DATA_READY and SPLIT_READY. All five remain search-blocked:
data completion does not resolve scientific evaluator/task-system,
aggregation, responsibility and output contracts. HotpotQA/HoVer retrieval
corpus and index remain separate unfrozen dependencies.

**BENCHMARK_DATA_FROZEN = NO** (suite incomplete).
**REAL_EXECUTION_READY = NO**.

See `test_summary.json`, `zero_api_audit.json`, `preflight.json`, and the
evaluator/isolation/privacy/source audit files for exact evidence and limits.
