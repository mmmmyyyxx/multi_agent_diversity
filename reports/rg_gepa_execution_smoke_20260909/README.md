# RG-GEPA production-path execution smoke

This zero-API fake-provider smoke supplements the durable-ledger repair before retry4. It exercises the production persistence and reconciliation helpers rather than a separate mock ledger implementation.

All three required interruption/retry scenarios pass:

1. A successful reflection is durable before a simulated case interruption.
2. One failed reflection attempt followed by success is audited as one logical call, two provider attempts, and one success.
3. A reflection followed by candidate Solver evaluation remains fully durable after a simulated interruption, and runtime Solver/optimizer accounting exactly matches the disk ledger.

Instrumentation is now frozen for a fresh retry4. This smoke made zero provider, Validation, and Test50 calls and did not read, resume, repair, or reuse retry2/retry3 incomplete evidence.

Verification: 22 focused tests passed; the canonical suite reported 946 passed with the single pre-existing historical-cache coverage failure unchanged. Compileall, governance preflight, sanitization, and diff checks passed.
