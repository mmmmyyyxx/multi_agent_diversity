# Layer-2 Pre-Pilot Repository Audit

This was intended as a zero-API scientific-boundary audit plus one narrow
contract repair. The audited base was `main@a80d57dc2ba31964c6320e95a4542a6f212cb396`.
No experiment, optimizer, Validation, or Test run was started. During an
over-broad full-suite invocation, two historical checkpoint tests inherited the
machine's real provider environment and each attempted one Solver request; both
failed authentication (401), returned no model output, and produced no
scientific evidence. This execution-protocol deviation is retained here rather
than reported as zero API.

## Outcome

`IMPLEMENTATION_READY_FOR_SEPARATELY_AUTHORIZED_INTEGRATION_CANARIES_WITH_AUDIT_API_ISOLATION_DEVIATION`

The unified runtime exposes all four combinations from one source tree:
`GEPA_NATIVE`, `GEPA_LAYER2`, `MARS_NATIVE`, and `MARS_LAYER2`.
Layer-2 GEPA uses a frozen packet schedule and fails closed instead of invoking
the native sampler or falling back after schedule exhaustion. Layer-2 MARS
evaluates exactly the packet's `local_eval_examples`; its native/global data
builder is not consulted. Existing poison tests exercise both boundaries.

Focus and anchor are outer-loop, one-step evidence. The transition store is
updated only by `SystemTeamCommitter.commit` after the candidate has a full
Optimize profile, its prompt identity matches, the active team is updated, and
the responsibility refresh and endpoint-identifiability persistence succeed.
Uncommitted local candidates cannot enter the next packet. The packet also
requires the stored transition child hash to equal the current parent prompt
hash.

## Contract defect found and repaired

The online assignment factory froze `local_validation_example_ids`, but the
Layer-2 packet builder ignored them and instead admitted every coalition row.
A deterministic reproduction showed one requested local-eval identity becoming
four actual identities. This violated the documented rule that Layer 2 freezes
the complete local curriculum before Layer 1 starts.

The repair makes explicit Layer-2 local-eval identities authoritative and
fail-closed if an identity is absent from the Optimize evidence universe. It
does not allow a backend/global-pool backfill. Direct packet fixtures that do
not supply explicit identities retain a deterministic Layer-2 coalition-row
compatibility path. The packet and selection-policy identities were versioned.

The packet now exposes and both backends persist sanitized pairwise role
intersection counts. No question, prompt, answer, provider endpoint, or raw
model response is stored by this telemetry.

## Frozen capacity and fallback semantics

- Online TeamMiniBatch remains the existing `4 responsibility / 4 coalition /
  4 preservation` contract. This audit did not change it.
- Responsibility evidence is required and is restricted to the primary lane.
- Root focus and anchor are empty; later focus/anchor are exactly the latest
  committed transition's newly-broken/newly-fixed sets.
- Explicit local-eval identities are used exactly, without filling from a
  global pool.
- Direct construction without explicit local-eval identities uses deterministic
  coalition rows selected by Layer 2.
- GEPA minibatch fill is the already-existing deterministic cyclic repeat over
  the frozen packet and is now named in packet provenance.

## Provider identity

Run/config identity records the logical provider profile and a one-way base-URL
identity hash. Preflight records presence and the hash, while call accounting
records the logical profile. Raw API keys and private endpoints are excluded.

## Verification

- Focused unified-boundary and API-isolation suite: `54 passed` (no provider
  requests). The two checkpoint tests responsible for the accidental attempts
  now inject an offline Solver and cannot inherit provider credentials.
- Broader two-layer suite: `54 passed`; five failures were pre-existing missing
  ignored/private historical artifacts, not assertion regressions.
- Full suite: `1159 passed`, `17 failed`, `13 errors`. Fifteen failures and all
  errors were missing ignored/private historical artifacts. Two additional
  failures were the unintended authentication attempts described above.
- Provider request attempts during audit: `2`, both failed before model output.
- Successful API calls: `0`.
- Validation calls: `0`.
- Test calls: `0`.

This audit does not establish scientific efficacy. The next allowed scientific
step is a separately frozen and explicitly API-authorized GEPA Layer2-feed
integration canary, followed later by the corresponding MARS canary.
