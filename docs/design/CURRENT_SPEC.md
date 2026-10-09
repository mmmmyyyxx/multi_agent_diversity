# Current Implementation Specification

The sole active research architecture is Unified Team Prompt Search.
The sole executable method is V2.4 Structured System Prompt Optimization:
`unified_team_prompt_search_v2_4_structured_system_prompt`.
[The V2.4 scientific contract](STRUCTURED_SYSTEM_PROMPT_V24.md) is normative.

New execution uses `scripts/run_experiment.py`, `MATHStructuredBinding`,
`CurrentPolicyBundle` and `build_current_team_prompt_search`. Each member owns
one immutable structured Role/Strategy/Answer prompt. Rendered System content
contains exactly those editable blocks. User content equals the raw question.
The five initial prompts are identical with independent realization lanes.
Natural ordinary response text is private observable evidence; written reasoning
is optional. The frozen gold-blind extractor and pinned mathematical equivalence
supply the sole binary reward. At most four semantic draws recover invalid output;
wrong parseable answers never trigger recovery and exhausted examples score zero.

Gradient compares response evidence and separately labeled Optimize worked
solutions, supports UNCERTAIN and advisory block suggestions. Layer1 edits one
block per generation with complete actual-diff lineage. The bounded two-level
search, independent Optimize roles, rotating preservation, initial competence
floor, target-or-team progress, equal plurality, fixed peers, winner-only Shadow,
transactional Memory, atomic commit and scientific stopping are preserved.
The [V3 transition contract](TRANSITION_TARGET_OR_TEAM_PROGRESS_V3.md) remains
normative. Six generations, four exports, 42 local evaluations and two Full
promotions remain fixed. No local acceptance veto gates outer admission.

Retired V2.1/V2.2/V2.3 optimizers and V6 instructions cannot execute from current
production. Historical manifests and reports retain their original scientific
meaning. Historical code recovery uses each frozen Git source. Shared scoring,
data governance, transport, persistence and accounting remain shared infrastructure.
No old realization, response cache, baseline, competence Memory or API scope is
eligible for V2.4. Fresh initial performance requires new real Solver requests.

The current development task authorizes zero API preparation only. Every future
paid attempt needs a fresh complete source/startup/manifest/request/cache/provider/
accounting freeze and exact single-use authorization. Validation is unauthorized;
Test is sealed. Canary reviews audit integrity without efficacy-based adaptation.
Readiness and synthetic conformance do not establish real-model performance.

Invariants: INV-UNIFIED-FLOW-001; INV-V24-STRUCTURED-SYSTEM-001;
INV-V24-GOLD-BLIND-RECOVERY-001; INV-V24-EDIT-LINEAGE-001;
INV-V24-FRESH-ATTEMPT-001. No new agent, critic, planner or team controller exists.
