# Protocol revision V1.1 — draft, blocked before execution

V1.0 was blocked before any real API call because historical V1 and V2 did not
have matched search-budget semantics. V1.1 removes historical V1 from the causal
factorial rather than modifying historical V1. No efficacy result was observed
before this revision.

The proposed 2 × 2 factorial is A1 (V2 Core), A2 (Pattern), A3 (Memory), A4
(Pattern + Memory), each on seeds 81, 82 and 83: 12 formal runs. There is no A0
causal arm. Historical V1 is NOT BUDGET MATCHED and NOT PART OF FACTORIAL.
V2_CORE_EFFECT = NOT_ESTIMATED_IN_V1_1.

Proposed estimands: P_simple=A2-A1; M_simple=A3-A1; P|M=A4-A3; M|P=A4-A2;
P_main=((A2-A1)+(A4-A3))/2; M_main=((A3-A1)+(A4-A2))/2;
I_PM=A4-A2-A3+A1. Primary endpoint is read-only post-freeze Validation VoteAcc.
Test remains sealed until the specified TEST_UNLOCK and frozen analysis.

The revised synthetic public-boundary budget gate passes for all four V2 toggles.
The model authority gate stops with STOP_MODEL_IDENTITY_CONFLICT. The current
defaults differ from the existing governed historical execution binding, and no
frozen Unified V2 experiment selects a formal model/provider binding. Historical
locks do not define V2 scientific authority; defaults cannot silently replace a
manifest-selected real execution. The Unified CLI itself still reports HOLD.

This is a registered DRAFT/HOLD proposal and a zero-API STOP audit, not a completed
preregistration or PREEXECUTION_FROZEN state. Initial team, real mechanism prompt,
finite provider ceilings, per-attempt authorization and execution hashes are
not frozen. Later gates and every real phase are NOT_RUN. No attempt is consumed.
No scientific source, historical V1, prior STOP evidence, data or prompts changed.
