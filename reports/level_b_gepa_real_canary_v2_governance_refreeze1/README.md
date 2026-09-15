# Level-B GEPA canary v2 governance refreeze

Gate: **PASS — AUTHORIZED MANIFEST, EXECUTION NOT STARTED**.

This refreeze changes only governance authorization metadata and source-freeze
hash semantics. The scientific implementation and the scientific protocol are
byte/semantic-equivalent to execution freeze `157f77a6c46790ff15b1e61a32ff0578f3ce297a`.

Managed UTF-8 text files now use `normalized_lf_text_v1`: CRLF, LF, and lone-CR
line endings normalize to LF before SHA-256. All other text bytes remain
significant, and binary files remain byte-exact. A separate clean checkout had
different raw protocol bytes but reproduced the normalized hash and passed the
complete source-freeze verification.

The manifest authorizes exactly one fresh stagefix2 retry. The fail-closed
manifest gate passes for the frozen Solver and reflection roles, but execution
still requires a later explicit user authorization and environment gate. No
formal run root has been created.

`API_CALLS=0`, `VALIDATION_CALLS=0`, `TEST_CALLS=0`.
