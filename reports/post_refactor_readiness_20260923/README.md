# Post-refactor readiness audit (zero API)

Source reviewed: `8d8855dc800f2e3a95006b60c8146424acfca8a4`.
This audit checks the architecture cleanup requested after the structural
refactor. It does not promote a scientific method or execute an experiment.

## Source-of-truth decision

The two-layer GEPA/MARS design is the active research direction. The
`member_aware_peer_state_v15` constant and checkpoint v25 remain the canonical
historical replay identity. `AGENTS.md` requires a separate versioned
promotion before changing the canonical scientific runtime. The opening of
`README.md`, `method.md`, `CURRENT_SPEC.md`, and `CURRENT_ARCHITECTURE.md` now
states this distinction directly. `docs/HISTORICAL_CODE_MAP.md` identifies
the present production code and retained replay paths.

## Run identity finding

`ExperimentSpec` previously hashed backend/scope/regime/configuration without
an explicit method version. It did not call `versions.METHOD_VERSION`.
However, the general `governance/provenance.py` builder still emits that v15
version, while `governance/manifest.py` accepts only the v15 method identity.
Those governance helpers therefore cannot simply be reused to publish a new
GEPA/MARS run. The typed spec now includes a distinct, versioned method
identity and the existing native/Layer-2 backend feed versions in its hash.
The v15 value and checkpoint identity remain unchanged.

## Pre-provider execution finding

The new CLI's earlier preflight accepted any non-empty `execution_factory`
string, and its execute path dynamically imported that factory before checking
a frozen startup bundle. A factory could have constructed a provider without
the existing authorization/lifecycle checks. Its preflight now reports
`HOLD: FROZEN_EXECUTION_GOVERNANCE_NOT_BOUND`; `--execute` aborts before any
factory import or provider construction. The deterministic engine and eight
offline golden paths remain callable through the typed Python API.

For a real post-refactor canary, bind one concrete execution adapter to source
hashes, preregistration, run identity, attempt-local authorization, provider
roles, lifecycle and ledger checks; then create a fresh freeze and obtain
explicit API authorization. The current CLI is not a real-provider launch gate.

## Historical tests

The pre-cleanup full suite was `1210 passed, 1 skipped, 35 failed, 20 errors`.
All 20 errors came from missing private `runs/` fixtures in the clean worktree.
Three module fixtures and one direct private-registry test now skip only when
their specific private inputs are absent. If those inputs are present, the
tests still execute and can fail normally.

The current full suite is `1212 passed, 23 skipped, 33 failed`. The 33 failures
must not be relabeled as missing-private skips wholesale: they include frozen
report hash mismatches on Windows with `core.autocrlf=true`, plus private
parent/replay files and historical identity assertions. For example, one
historical report expects LF bytes for its README and text reports while the
current checkout materializes them as CRLF; the same report expects CRLF for
other files. The repository needs an explicit, report-specific byte policy and
case-by-case fixture audit before default `pytest` can honestly be green.
No historical report or run artifact was rewritten here.

## Current gate

| Check | Status |
|---|---|
| Two-layer engine golden semantics | PASS (8/8) |
| Active direction versus v15 replay documentation | CLARIFIED |
| New method identity distinct from v15 | PASS |
| CLI real-provider startup governance | HOLD, fail closed |
| Default full `pytest` | NOT GREEN (33 historical failures) |
| Real post-refactor GEPA Layer-2 canary | NOT AUTHORIZED / NOT RUN |
| Provider / Validation / Test calls in this audit | 0 / 0 / 0 |

The next work should bind the CLI to a concrete frozen execution adapter and
resolve historical test fixtures and checkout-byte rules. Scientific GEPA,
MARS, Layer-2, admission and stopping definitions were not changed.
