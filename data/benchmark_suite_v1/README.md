# Benchmark Data Freeze V1

Only IFBench currently has an actual frozen dataset and ordered memberships.
The user deferred the remaining public data downloads. `download_plan.json`
contains **proposed** sources and counts, not proof that those data exist.
`suite.json` is created only when all five actual dataset freezes succeed.

## Download later

Create an isolated Python environment and install
`materializer-requirements.txt`. The task used Python 3.13.5. No provider or
optimizer dependency is needed by the materializer. Public HTTP downloads do
not authorize any model evaluation.

```powershell
python -m venv data/benchmark_suite_v1/raw/_environment
data/benchmark_suite_v1/raw/_environment/Scripts/python.exe -m pip install -r data/benchmark_suite_v1/materializer-requirements.txt
data/benchmark_suite_v1/raw/_environment/Scripts/python.exe scripts/materialize_benchmark_suite_v1.py --all --download
```

Alternatively select `--benchmark hotpotqa`, `hover`, `ifbench`, `pupa`, or
`math` with `--download`. Both selection and mode are required. The existing
isolated environment under ignored `raw/_environment` can also be reused.

Downloads use the pinned revisions in `benchmarks/data_freeze.py`. HoVer's HF
loader points to mutable GitHub data; this recipe additionally pins the data
commit and reproduces the inspected loader's row projection. MATH uses the
canonical seven subject configurations of `EleutherAI/hendrycks_math` after
the preferred official archive rejected anonymous access and the qwedsacf
mirror did not expose original train/test membership. That source recipe has
not yet been downloaded or validated. It requires actual 7,500/5,000 counts
before freezing the explicitly proposed project split.

Existing source caches require a matching URL/SHA256 receipt. The downloaded
HotpotQA first shard and both IFBench files already have local receipts. Do
not insert arbitrary files or fabricate receipts to satisfy this check. New
downloads save receipts automatically. Existing manifests, memberships, raw
canonical JSONL and materialized splits cannot be overwritten with different
bytes. A mismatch fails closed.

## Verify without network

```powershell
python tests/formal_zero_api_runner.py --offline-command scripts/materialize_benchmark_suite_v1.py --benchmark ifbench --verify-only
# Once all five downloads and freezes exist:
python tests/formal_zero_api_runner.py --offline-command scripts/materialize_benchmark_suite_v1.py --all --verify-only
```

Verification checks source bytes, canonical row schemas/hashes, ordered
membership, exact split reconstruction, materialized files and suite hashes.
It never downloads a missing file. Raw datasets, caches, materialized examples
and environments stay ignored. Membership files contain identifiers/hashes;
PUPA uses hashed identifiers and never stores raw query/PII/response values.

## Access and readiness

Future experiment composition must bind `FrozenSplitReader` to the expected
manifest SHA256. Mutation, reflection, evidence, pattern and memory use only
search data. Shadow raw rows are available only to `ADAPTIVE_GATE`; the value
returned to search is an `AdaptiveGateFeedback` score/pass-fail object. Test
and independent validation raw access is forbidden during search. Test
manifests/hashes remain readable. V1 independent validation is empty.

Downloading or freezing data does not authorize search and does not freeze an
evaluator, task system, aggregation, responsibility or output contract.
Registry flags record evidence actually completed in this source revision;
later downloads do not automatically promote a benchmark to search-ready.
For now **BENCHMARK_DATA_FROZEN=NO**, **REAL_EXECUTION_READY=NO**.
