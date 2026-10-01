# MATH + IFBench + HotpotQA data/split migration

```text
BASE_SHA = 19d0cbb38277ec0dde32e4df9b621be0364f0931
DATASET_SUITE = MATH + IFBench + HotpotQA
BBH_ACTIVE_NEW_EXPERIMENT = NO
REAL_LLM_CALLS = 0
DATA_DOWNLOADS = YES
FORMAL_EFFICACY_RUNS = 0
```

The user explicitly amended the baseline to local HEAD. origin/main was
a5997898c30c4d4daad5511203657d3c687c90e5 at the verified fetch; this task
makes a local commit only and grants no push or model-call authorization.

Canonical freezes and separate experiment memberships are immutable. IFBench
old canonical bytes and Optimize/Shadow/Test identities are preserved. HotpotQA
project Test is official labeled validation. MATH membership exactness is
PROJECT_PROPOSED; no AgentGrad exact split reproduction is claimed.

| Benchmark | Optimize | Shadow | Validation | Test | Isolation |
|---|---:|---:|---:|---:|---|
| MATH | 150 | 300 | 300 | 300 | DISJOINT |
| IFBench | 150 | 300 | 300 | 294 | DISJOINT |
| HotpotQA | 150 | 300 | 300 | 300 | DISJOINT |

All members, arms, seeds and adaptive re-evaluations bind Solver qwen3-8b.
Optimizer/reflection and Pattern bind qwen3.7-flash. Neither CLI defaults nor
data freeze confer authorization. Raw data, resources, environments and model
assets remain ignored; only memberships, hashes, manifests and sanitized
evidence are published locally.

IFBench checker runtime is frozen in an isolated environment; scientific
aggregation-aware responsibility remains HOLD. HotpotQA retrieval remains HOLD.
MATH needs benchmark-specific prompts, V2 composition and governed execution
binding before any real preexecution/canary. Pattern/Memory are default off.

MATH distribution includes one upstream Level ? Validation row, retained.
No resampling follows descriptive distribution audits.

## Requested ordered task fields

| # | Field | Value / evidence |
|---:|---|---|
| 1 | BASE_SHA | 19d0cbb38277ec0dde32e4df9b621be0364f0931 |
| 2 | local commit SHA | See containing local Git commit; self-reference excluded |
| 3 | changed files | 66 task files; exact paths in final_report_fields.json and containing Git commit |
| 4 | BBH current status | HISTORICAL_DEVELOPMENT_REPLAY_ONLY |
| 5 | active benchmark suite | ["math", "ifbench", "hotpotqa"] |
| 6 | MATH source revision | 21a5633873b6a120296cce3e2df9d5550074f4a3 |
| 7 | MATH train/test counts | {"train": 7500, "test": 5000} |
| 8 | MATH raw source hashes | {"train": "fb770320beb0ab114fe50b5b1a7fbc93a0f7adefd91159db4ff33cd9d4f8058b", "test": "c229f4e8194706dbf8580ad2a127cfc3fdc827e824396e0e220eff76f754ae8a"} |
| 9 | MATH canonical freeze status | FROZEN |
| 10 | MATH Optimize/Shadow/Validation/Test counts | {"optimize": 150, "shadow": 300, "test": 300, "validation": 300} |
| 11 | MATH membership hashes | {"optimize": "07736245014e15c4a1b3f6365794b9bfd25d58032ee04fb0578731c6bf23fec9", "shadow": "80d238e21644777f348694e2df3a93af395abbca648fb0e4e3f0c01211be3667", "test": "9cba0605a224f1c1ebe15a01c65d1263f8cbee45d8ec2cfd67d496b119a6c32d", "validation": "1016db4741d49bc8c965b6dbc5bfa55c30f5a4fa77165e70494e6f6cd0d698b3"} |
| 12 | MATH overlap audit | DISJOINT; all six pairs and all three hash types |
| 13 | IFBench canonical manifest hash | 7076ebb0dac504fd279093043dd3851be0d2d577af8d02a530b08a7840181e2d |
| 14 | IFBench old Optimize parity | BYTE_MEMBERSHIP_ID_SEQUENCE_UNCHANGED |
| 15 | IFBench old Shadow parity | BYTE_MEMBERSHIP_ID_SEQUENCE_UNCHANGED |
| 16 | IFBench old Test parity | BYTE_MEMBERSHIP_ID_SEQUENCE_UNCHANGED |
| 17 | IFBench new Validation count/hash | {"count": 300, "hash": "70415c0bad3720cfc2beadec04f80601e4c8fed5a0e27eff9a5c9aba8eb7d827"} |
| 18 | IFBench four-way overlap audit | DISJOINT |
| 19 | Hotpot source revision | 1908d6afbbead072334abe2965f91bd2709910ab |
| 20 | Hotpot train/validation counts | {"train": 90447, "validation": 7405} |
| 21 | Hotpot source hashes | {"train": "2a98eeac97dfc79f6ca5a5d7748bd500a1e63b0a89aaa3ed592d8a7eaba6bff6", "validation": "bd0e39459a3fe2cade76ad1744fc3887ff9664e8fcc9466303c9d3c5e481e6f8"} |
| 22 | Hotpot canonical freeze status | FROZEN |
| 23 | Hotpot Optimize/Shadow/Validation/Test counts | {"optimize": 150, "shadow": 300, "test": 300, "validation": 300} |
| 24 | Hotpot membership hashes | {"optimize": "799f0edcac9cdec67a10cf86f64fc4387811d705e7300b8382821f2c481deeea", "shadow": "8fbafd859b76adaa78a4806fe41e8f62f364eec8cb3d38386d5f27634351f6d0", "test": "f024aae08b5f4dd15f9805635563ff256004d97ff25c27d43e87693f4ef53ed6", "validation": "64c5304bf014cb19ec2b10e27aacad49ffeb35f21b1e733abd060cd68822f983"} |
| 25 | Hotpot overlap audit | DISJOINT |
| 26 | MATH evaluator readiness | YES; pinned math-verify 0.6.0 isolated dependency environment |
| 27 | MATH responsibility readiness | YES; binary plurality unchanged |
| 28 | MATH system readiness | YES; execution binding still pending |
| 29 | IFBench evaluator dependency readiness | YES; isolated lock verified with positive/negative and byte-poison controls |
| 30 | IFBench responsibility readiness | NO; RESPONSIBILITY_POLICY_NOT_FROZEN |
| 31 | Hotpot retrieval readiness | NO; SYSTEM_DEPENDENCY_RETRIEVAL_NOT_FROZEN |
| 32 | Hotpot responsibility readiness | YES; normalized-answer binary plurality unchanged |
| 33 | three-benchmark readiness matrix | benchmark_readiness_matrix.json |
| 34 | downloaded bytes | {"math": 4883857, "ifbench": 0, "hotpotqa": 194204299} |
| 35 | dataset HTTP attempts | 16 |
| 36 | LLM provider attempts | 0 |
| 37 | raw data tracked by git? | NO |
| 38 | materialized rows tracked? | NO |
| 39 | secret scan | PASS; sanitization_audit.json |
| 40 | tests passed/skipped/failed | 510 passed / 2 skipped / 0 failed; 1468 historical cases deselected |

## Final readiness

```text
ACTIVE_BENCHMARK_SUITE = MATH + IFBench + HotpotQA
BBH_ACTIVE_NEW_EXPERIMENT = NO
MATH_DATA_READY = YES
MATH_EXPERIMENT_SPLIT_READY = YES
IFBENCH_DATA_READY = YES
IFBENCH_EXPERIMENT_SPLIT_READY = YES
HOTPOTQA_DATA_READY = YES
HOTPOTQA_EXPERIMENT_SPLIT_READY = YES
REAL_LLM_CALLS = 0
FORMAL_EFFICACY_RUNS = 0
SCIENTIFIC_EFFICACY_VERIFIED = NO
SOTA_VERIFIED = NO
REAL_EXECUTION_READY = NO
```

The source SHA, canonical manifest SHA, experiment manifest SHA and scientific
contract SHA are separately recorded in source_identity.json. Report hashes
are listed in sha256_manifest.json; the containing Git commit records publication.

## Verification completion

Current classified suite: **510 passed, 2 skipped, 0 failed**; 1468 historical cases deselected. No full historical private-artifact replay claim. Compileall, canonical/experiment verify-only, governance/lineage/manifest preflight, sanitization and git diff --check passed. Pre-import guard reports zero runtime network attempts. IFBench isolated checker/resource positive/negative checks and byte-poison rejection passed. No models were called.
