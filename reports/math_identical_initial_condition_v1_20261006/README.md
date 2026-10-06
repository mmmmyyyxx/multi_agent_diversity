# Identical minimal MATH initialization

V1_1 → V1_2 is an initial-condition correction. The scientific method is unchanged.
Five members (IDs 0–4) now start from the byte-identical minimal mutable prompt
`Solve the problem.`; unique initial prompt count changes from 5 to 1. No role,
reasoning strategy, verification behavior, specialization or diversity instruction
is predefined. Independent member identities and Solver request/cache lanes remain.

- Prompt SHA256: `4e6dfb1595690c9487e7dea0deaafa72363cde09f5ce81186f1e4fbdc4205f80`
- Ordered team SHA256: `d1a04dbdd540371638e8271cca90469f0a8cf4d42a5f944eeee8c6dfcfca9110`
- Initial artifact SHA256: `bdde9df785adaaebcdfcafcc6b5c39f461be072e0cd31b6026f8925460b69cf2`
- Runtime source: `52ada98b4735cfa7713de2c2f5437a94ce0bef0b`

The current validator enforces five distinct ordered IDs, one exact minimal prompt,
NONE data dependency, valid mutable prompt contract and internal/file/team hashes.
Five identical wire requests dispatch through distinct member identities; same-member
exact requests reuse both in-memory and durable caches. Fresh attempts produce new
identities and cannot reopen the old durable cache. Fake-provider full compositions
also exercise current Pattern, Memory and Layer1 behavior from V1_2.

Complete current coverage after fixture repair: **1314 passing cases,
2 skipped, no unresolved failure**. The complete suite invocation
recorded 1312 passed, two old initialization-fixture
failures and 2 skipped. Both failures were repaired in test fixtures;
the affected Rolling Memory module then passed all 41
cases. Production source did not change. Final initialization/runtime and governance
recheck: 145 passed, zero failures. These counts combine the complete suite
and affected-module recheck; they do not describe a single all-green full invocation.
Targeted conformance included 219 passing cases and the final six fixture/helper
regressions. Compileall, governance, manifest/preflight, dependency isolation,
hash preservation, sanitization and git diff checks pass. All verification uses
credential-free processes with the network guard active before application imports;
test-process network attempts and real provider calls are zero. Initial fixture
failures were repaired in test infrastructure; the incomplete initial full-suite
invocation was stopped and superseded by the complete final-source suite.

Historical MATH compatibility cases explicitly use a copied V1_1 artifact
workspace. Legacy implementation remains unchanged. Historical private-artifact
tests and full historical replay were not executed, and no complete historical
replay pass is claimed. Existing Canary/Pilot attempt1 bindings, manifests,
reports and runtime journals retain their original bytes. The old treatment was
five similar but textually different prompts; the new treatment is five identical
minimal prompts. They must not be described as the same initialization treatment.

The archive and invariant index retain exact old bytes and source provenance.
Only an explicit initial-condition amendment resolves the old path/hash pair for
parent receipt verification. All other dependencies retain exact integrity checks.
New offline bindings use fresh attempt/cache identities and V1_2 artifact/team
hashes. Old Pilot authorization remains in the immutable parent receipt and is
removed from the new effective binding; it grants no new execution authority.

Responsibility, Gradient prompt/guard/clustering, Pattern F, Memory, Layer1,
TeamProbe, Full, Shadow, models, decoding, ceilings and dataset memberships remain
unchanged. The numeric-guard issue from Pilot attempt1 is not changed.

**REAL_PROVIDER_CALLS=0; CANARY=NO; PILOT=NO; VALIDATION=NO; TEST=NO.**
The new manifest remains DRAFT and preflight remains HOLD/PREEXECUTION_NOT_FROZEN.
Future real execution requires another fresh attempt/binding/cache namespace,
empty Memory, fresh Solver lanes, source/protocol freeze and exact single-use
authorization. No experiment is rerun and no scientific efficacy is inferred.

Machine-readable evidence: [summary](summary.json), [verification](verification.json),
[provenance](provenance.json), [engineering audit](engineering_audit.json).
