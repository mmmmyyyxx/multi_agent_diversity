# Online GEPA local-to-team transfer diagnostic — zero-API freeze

Status: `READY_FOR_AUTHORIZATION`. No real run has started. This report freezes
the prospective Seed81, single-trajectory diagnostic described in
`experiments/gepa_layer2_local_to_team_transfer_diagnostic_v1/PROTOCOL.md`.
It is **not** a canary result, a local-to-team transfer result, or permission
to execute. A separate one-time API authorization for this exact attempt is
still required.

| Identity | Frozen value |
| --- | --- |
| Execution source | `b3b428419b07badf75e2b1fbbbcdb0d8475e8dae` |
| Attempt | `gepa_layer2_local_to_team_transfer_diagnostic_v1` |
| Preregistration SHA256 | `216ef7f9251d8670e22671e90876b4c65dccb401606bc3f33245dde535172d16` |
| Protocol SHA256 | `27f3d3fa3759fdbf4de2fa1f5eb23325d454fd5b5b653730644a812c3a2f6652` |
| Run identity SHA256 | `65c3d4fdb0dd47581f90b3af446cb0ba4839c73d41275583d2ba8cbe208123c4` |
| Provider / Solver / Reflection | `lwj` / `qwen3-8b` thinking false / `qwen3.7-flash` |
| GEPA | pinned v0.1.1, `b4dbb55b7601dac448cdb836d5a401ca7d9eb920` |
| API / Validation50 / Test50 calls | `0 / 0 / 0` |
| Authorization | `false` |
| Formal run root | absent |

The private, ignored preparation bundle contains source/data/provider hashes
and the frozen execution command inputs. The official preflight returned
`PREREGISTERED_NOT_EXECUTED` and `ready_for_authorization=true`. It did not
construct a provider client or create a formal run root.

The code path preserves ordinary MiniBatch → Common-Safe → winner-only Shadow
→ write-back. Mandatory Full for a MiniBatch failure is isolated under
`diagnostic_full_eval`, never attached to the online candidate record used by
the selector. An internal GEPA acceptance/returned-frontier mismatch fails
closed; it is not silently omitted from the sample. The dynamic opportunity
source uses the actual committed state, so repeated `S0` parents remain
visible if no genuine team commit occurs.

The zero-API cost upper envelope is 500 initialization Solver calls, up to 360
local Solver calls, 20 Reflection proposals, 500 unique candidate-profile
Solver calls for five MiniBatch+Full pairs, and up to 1,500 optional ordinary
Shadow Solver rows: unconstrained upper envelope 2,880 successful calls. The
hard emergency ceilings are 1,200 successes and 4,800 transport attempts.
Thus the five-mutation target is **not guaranteed** before emergency stop;
an incomplete emergency result cannot be reported as five complete Full
measurements. See `cost_envelope.json`.

Validation used zero real API: 90 focused tests passed; `compileall`, frozen
preflight, source/hash replay, sanitization scan, and `git diff --check` passed.
The full historical suite produced 1,280 passed, 1 skipped, 15 failed and 13
errors; every non-pass is from previously missing private historical artifacts,
not this diagnostic's tests. The unrelated untracked historical retry2 report
was left untouched and excluded from the commit.

No push was requested. If execution is later authorized, use a clean checkout
of the **exact execution source commit above**. Do not substitute this report's
later commit for the execution source, regenerate the prep, change its
authorization scope, or reuse an aborted run without a new freeze and grant.
