# Transfer diagnostic v2: engineering abort and zero-API repair

Classification: `PILOT_ENGINEERING_FAILURE / SCIENTIFIC_NON_RESULT`.

The single authorized real-provider attempt reached `RUNNING` and then
`ABORTED` during initial fixed-probe profile persistence, before local GEPA
search or any TeamMiniBatch, Full, Shadow, or write-back decision. The durable
ledger has 100 unique initialization Solver records: 100 provider attempts,
100 successes, and no failures. Validation50 and Test50 were not accessed.
The original run and consumed authorization remain preserved and are not
eligible for resume or reuse.

The demonstrated fault was Windows path-length handling for an atomic JSON
replacement. The destination was 261 characters, its parent directory
existed, and the unprefixed `os.replace` raised `WinError 3`. The repair adds
the Windows extended-path prefix only at the atomic JSON replacement boundary;
non-Windows behavior and all scientific methods, models, data, GEPA budgets,
gates, and stopping rules remain unchanged.

Verification: targeted long-path and online-transfer tests `25 passed`;
full suite `1297 passed, 1 skipped, 15 failed, 13 errors`. The nonpassing
tests are historical private-artifact dependencies and match the pre-repair
categories. `compileall`, governance preflight, and `git diff --check` passed.

The repaired source is `740ebfd54a9e6f8825934ef6c8dd20a69322b0b1`.
A fresh, **unauthorized and unexecuted** preparation at
`runs/gepa_layer2_local_to_team_transfer_diagnostic_v2/prep_retry1` has:

- preregistration SHA256:
  `246ef5edb460711ff61687095d7fe85a68689d8a9b31807b9c6e63fbcde79f74`
- run identity SHA256:
  `5169234de36901bbec9c92fabd8adf0d7151727814598e2fed310eb0f8a9f6c2`
- state: `PREREGISTERED_NOT_EXECUTED / READY_FOR_AUTHORIZATION`
- provider attempts, Validation50, Test50: `0 / 0 / 0` for this new prep.

The logical experiment identifier remains v2, but the new source and run
identity are distinct from the aborted attempt. No new real run root was
created. This preparation does not itself authorize another API call. A fresh
explicit authorization and exact-source launch check are required before any
execution. Formal v3 preparation is deferred until this diagnostic completes
operationally; negative scientific outcomes will not trigger another pilot.
