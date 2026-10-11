# Optimizer output contract and bounded feedback V1

Normative for `unified_team_prompt_search_v2_7_output_contract_feedback_v1`.
This amendment inherits the direct readable evidence, concurrency and timing
contract and replaces the optimizer output instructions and recovery feedback.
It requires a fresh method, binding, cache, source and phase authorization.
Earlier sources, manifests, trajectories and outcomes remain immutable.

Gradient output has exactly six keys. `suggested_block` is one role, strategy
or answer string, or JSON null; it is never a mapping or a list. ACTIONABLE
requires nonempty observation, diagnosis, correction and effect strings.
UNCERTAIN retains nonempty observation/diagnosis and null correction/effect.
Decoded character limits remain 400/400/400/240. Correction abstraction,
source provenance and leakage checks remain separate and mandatory.

Reflection returns exactly the existing NO_SAFE_EDIT object, or the existing
four-key PROPOSE_EDIT object. `target_block` is one string from the same three
blocks. The complete one-block replacement, 240-character summary and
3000-character rendered prompt bounds remain. Generic valid shape examples
and JSON escaping instructions describe types without imposing a block choice.
No provider JSON-schema capability is assumed or silently enabled.

Gradient output rejections and Reflection format rejections record a source-free feedback object:
identity, stage, category and at most six issues. Each issue has only a closed
field/code/type, actual character count and limit. Rendered prompt length is
explicitly labeled rendered_system_prompt, not replacement-block length.
No rejected text, extra
model-generated key, question, gold, reference span or hidden reasoning enters
feedback. Next draw receives only the most recent rejection. Gradient puts it
in the separately labeled output_contract_feedback field; Reflection places
the labeled metadata before the full unchanged task evidence. It is format
feedback, never a mathematical repair instruction or shared Memory item.
Valid output clears pending Reflection feedback. Integrity or provenance
exceptions remain fatal and do not trigger regeneration.
The existing bounded optimizer-dependency check recognizes the new reserved
output_contract_feedback identifier before a candidate can reach Solver.
This extends known-field coverage of the existing optimizer-context prohibition;
it is not a semantic proof of all possible disguised dependencies.

Recovery remains bounded at three Gradient draws per wrong example and six
total Reflection generations per opportunity. First valid ACTIONABLE or
UNCERTAIN stops Gradient recovery. Rejected text is never repaired, truncated,
accepted or reused. Reflection format failures continue to consume one of six
generations. Unknown exceptions remain fatal. JSON parsing and substantive
acceptance guards remain strict; new diagnostics describe existing rejections.

Solver16, Gradient8, serial per-example recovery, all-gradient-drain barrier,
serial Reflection and team transitions, private request timing, visible evidence,
split boundaries, model generation settings and resource ceilings remain.
Pattern clustering, Layer1 evaluation/ranking, Full/Shadow admission and scientific
stopping remain. Retry request bytes now differ because they contain feedback;
this is an explicitly versioned generation behavior change, not efficacy proof.

Offline tests can verify schema, routing, bounds and evidence retention only.
They cannot establish better model compliance, reduced cost or mathematical
efficacy. The preparation manifest is DRAFT, unauthorized and unbound. No old
scope, cache or checkpoint is eligible for current execution.
