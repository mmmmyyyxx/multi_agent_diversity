# Independent single-Prompt Validation100 gated preparation

This is an offline implementation of the user-requested mechanism amendment,
not a live efficacy result. The [protocol](../../docs/design/SINGLE_PROMPT_VALIDATION_GATED_V2.md)
and [manifest](../../experiments/manifests/single_prompt_validation_gated_v2.json)
define fresh three-arm, three-seed, twenty-opportunity trajectories.

Each opportunity allows one initial generation plus at most three retries for
malformed, overlong, unchanged or otherwise illegal output. Four failed responses
abandon that opportunity, consuming one of twenty and preserving the Prompt and
both recorded profiles without scoring a fallback. Abandonment is recorded separately
from a legal candidate's gate rejection. A legal candidate is evaluated on Optimize150 and
the common frozen Validation100. Retain only if it answers at least one more
Validation question correctly than the retained recorded profile. Legal equal
or worse candidates are rejected and complete the opportunity; rejection keeps
the incumbent Prompt and both recorded profiles. No Optimize acceptance gate.

Sampling uses existing Validation300 membership metadata, uniform without
replacement, fixed Seed88 and original order. Training authorization must explicitly
include these100 adaptive-gate items. The later restart request also grants automatic
final Test300 after all nine training cells close and their sealed evidence passes
integrity/accounting checks. Test uses a distinct derived scope and ledger, with
the exact300 formal MATH members and matching Solver/scoring/invalidity contracts.
The stopped V1 scopes and evidence are preserved. V2 starts from the common initial
state and requires a new source/manifest freeze and exact execution grant. This
request grants fresh V2 execution; it does not resume the stopped V1 state.

Verification uses synthetic arithmetic fixtures and a credential-free pre-import
network guard. It never opens actual Validation/Test content. Detailed final check
counters are recorded after the checks complete. Cumulative Tokens have no hard
ceiling; finite physical-call ceiling190440 includes all timeout/format retries.
Exhaustion preserves an unfinished opportunity rather than completing a fallback.
The report itself grants no real execution or efficacy conclusion. The exact private
execution grant records the subsequent human restart and conditional final-Test request.
