# Current Implementation Specification

The sole active research architecture is Unified Team Prompt Search.
**V2.3 is the sole active scientific method. V2.2 is permanently retired.**
This specification and the linked component contracts are design authority;
reports and historical parent manifests are evidence only.

New execution uses `scripts/run_experiment.py`, `MATHEvidenceBinding`,
`CurrentPolicyBundle` and `build_current_team_prompt_search`.
The only admissible method/binding pair is `unified_team_prompt_search_v2_3`
and `MATH_OPTIMIZATION_EVIDENCE_BINDING_V1`. Complete optimization-evidence
and V6 trajectory policies are mandatory. Missing, historical, paired or
unknown runtime identities fail before data/provider/cache construction.
`versions.py` preserves historical identifiers without execution eligibility.

## Scientific graph and preserved behavior

One `UnifiedSearchOrchestrator` owns state analysis, opportunity selection,
Layer1, fixed-peer evaluation, transition, winner-only Shadow, atomic commit,
history and stopping. The [V2.3 evidence contract](OPTIMIZATION_EVIDENCE_V23.md)
is normative for exact sampling, search, hypothesis and measured Memory rules.

Five qwen3-8b members use the existing five identical generic procedures,
thinking disabled, the frozen Solver sampling and V6 ordinary visible solution
interface. qwen3.7-flash supplies Gradient, clustering and mutation.
The evaluator scores only the strictly extracted final answer with unchanged
mathematical equivalence. Provider-private reasoning is not an evidence source.
Equal-weight equivalence plurality, fixed peers and one-member updates remain.

Member responsibility and Pattern responsibility retain their existing raw
`max(4D,2N,C)` primitives; member allocation discounts prior failures by `1+f`.
Layer2 selects WHO, Pattern and assigned examples before Layer1 reads Memory.
No unrelated team/controller metadata enters the mutation model.

Every incorrect selected-member Optimize example gets one independent diagnostic
with the actual member procedure and visible execution plus a separately labeled
dataset reference solution. UNCERTAIN is legitimate; only actionable corrections
enter clustering and unchanged Pattern selection. Pattern is a repair hypothesis.
Layer1 proposes a small standalone edit or `NO_SAFE_EDIT`.

Mutation, SearchValidation and TeamProbe each use three frozen, pairwise-disjoint
Optimize examples. Correct preservation examples rotate deterministically by
seed/member and use current committed coverage. Validation examples/answers
never enter pre-generation mutation requests; parent and child share membership.
Six generations yield at most four exports, 42 logical local Solver evaluations,
12 Probe evaluations and two Full promotions. No local survival veto gates
outer admission. Full measures the entire Optimize membership.

The [V3 transition contract](TRANSITION_TARGET_OR_TEAM_PROGRESS_V3.md) remains
unchanged: retain immutable initial member competence, nonregressing team Vote,
and positive target or team progress. Ranking remains team gain then target gain;
local/soft scores do not replace Full deployment criteria. Only the winner is
evaluated on Shadow. Failed gates cannot change committed state.
Scientific stopping remains team-epoch no-commit patience two. Operational
resource/horizon limits are separately frozen and do not imply saturation.

Initial Optimize profiles bootstrap five measured competence records without
extra calls. Edit-effect Memory binds actual diffs and parent/candidate hashes,
local/Probe/Full scopes and committed status. Full rejection produces measured
negative evidence; candidate gains never replace committed coverage. Existing
private/shared storage, retrieval, risk promotion and 1200-character limits remain.
Memory is prepared/validated transactionally and applied only after state handling.

## Invariants and execution governance

- **INV-UNIFIED-FLOW-001**: One production graph and one scientific owner.
- **INV-MATH-VISIBLE-SOLUTION-001**: Observable ordinary solution evidence,
  exact provenance, no fabricated or hidden reasoning, final-answer-only scoring.
- **INV-V23-ONLY-CLOSURE-001**: Historical constructors/defaults/bindings cannot
  enter new execution. Old implementation snapshots are non-importable provenance.
- **INV-V23-EVIDENCE-001**: Independent Optimize roles and post-generation checks.
- **INV-V23-MEMORY-001**: Bootstrap measured competence; update coverage on commit.
- **INV-V23-AUTHORIZATION-001**: Fresh frozen attempt, source, hashes, request,
  cache/accounting scope and exact single-use user approval before any paid call.

The [single-arm execution contract](V23_ONLY_EXECUTION_SCOPE_V1.md) separates
Canary verification from scientific decisions without changing search. No V2.2
or A/B execution is authorized. Validation remains unauthorized; Test sealed.
Historical 40M allowance and cancelled paired approvals grant no new authority.
Operational repair must preserve evidence and re-freeze affected identities;
scientific or ambiguous repairs require an explicit user decision.

Previous normative bodies and runtime blobs are preserved verbatim in the
[source archive](../archive/runtime/a4_pre_v23_only_7342d85/README.md).
Historical replay requires the original frozen Git checkout and authorization.
Shared mathematical scoring, transport/accounting physics and schema validation
may retain their original names; they cannot choose a historical optimizer.
