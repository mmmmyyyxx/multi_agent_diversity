# MATH Baseline Stage 1 Preparation

The user accepted the previously proposed Stage 1 API scope. This preparation
implements its independent, governed A/B/C runner. No model call has been made
in this publication. Stage 2, Shadow, Validation, Test and Unified optimization
remain outside this scope.

The immutable design protocol and membership from the baseline audit are retained:
60 Optimize examples, 180 logical evaluations, qwen3-8b, thinking=false, seed81,
one Solver worker, identical native A/B Parser, separate strict boxed C Parser,
four invalid-only semantic draws, up to 21 physical attempts per draw and the
immediately-previous-truncation capacity rule (3600/6144).

The independent fresh accounting ceiling is 1,450,000 charged plus reserved
tokens. The fsynced hash-chain, exclusive consumed-scope marker, immutable
receipts and sealed checkpoint prevent old allowance or evidence reuse and
silent paid retries after persistence errors. SDK retries remain disabled;
connection, timeout, rate-limit and server errors follow the original bounded
transport policy. This is an operational bound, not a provider billing claim.

Scientific and execution policy: docs/workflows/MATH_BASELINE_STAGE1_EXECUTION_V1.md.
Preregistration: experiments/manifests/math_baseline_calibration_stage1_v1.json.
The source implementation is committed first. A private frozen manifest then
binds exact clean main HEAD, source hashes, protocol, selected rows, interpreter,
dependency identities, provider fingerprint, exact command and the contextual
user authorization. The governed handoff is verified before the first HTTP call.
The earlier audit and design publications remain immutable.

Offline verification is recorded in verification.json after the final current
suite, compileall, governance, deterministic fake HTTP/ledger/recovery tests,
sanitization and diff checks. Historical/private artifact modules remain omitted
according to tests/suite_classification.json. No historical replay PASS is claimed.
No push is included in the API authorization.
