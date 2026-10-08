# MATH visible solution feedback V1

This is an explicitly bound Solver execution and feedback-interface correction
within Unified Team Prompt Search V2.2. It changes provider-visible instructions
and optimizer evidence, so scientific/runtime/request identities change. It does
not redesign the search algorithm or establish efficacy.

`MATH_SOLVER_INTERFACE_V6` requests a clear, logically ordered mathematical
solution in ordinary assistant content, ending in exactly one `FINAL_ANSWER:`
line. Its nonempty mathematical payload is on that line, which is the last
nonempty line; nothing follows it. The immutable instruction supplies formatting
and visibility requirements, not a domain-specific solving strategy. There is
no answer-only user suffix. The five minimal mutable initial prompts are unchanged.

The final boundary parser, payload parser, pinned mathematical equivalence
evaluator, invalid-output classification, four-attempt recovery, plurality and
team scores are unchanged. Only the extracted final payload is scored. Reasoning
text is never a fallback answer. A missing written solution is explicitly marked
but does not add a correctness/validity penalty. A truncated response remains
invalid even if it contains an otherwise valid final line.

Solver remains qwen3-8b with the exact `SOLVER_DECODING_POLICY_V2`, including
thinking=false and the existing 3600-token output cap. One existing Solver request
produces the written solution and final answer. Provider-private reasoning fields
are not used as written solutions; no additional model, request or provider mode
is introduced.

`MATH_VISIBLE_SOLVER_PROFILE_V1` retains the complete resolved prediction,
including every original invalid/recovered response. Its private feedback
projection, `MATH_VISIBLE_SOLVER_TRAJECTORY_V1`, binds member slot, mutable prompt
hash, public input hash, example ID, split, actual request/cache identity, raw
response hash and complete prediction identity. State-to-evidence conversion
checks those bindings against the selected member, current procedure and input.

`MATH_VISIBLE_SOLUTION_FEEDBACK_POLICY_V1` exposes a deterministic Unicode prefix
of at most 4096 visible-solution characters per observation. Original/exposed
lengths, feedback truncation, response truncation and solution status are explicit.
Statuses distinguish written solution present, written solution missing, invalid
final boundary and truncated response. With an invalid boundary the response is
explicitly unsegmented/incomplete evidence; no final answer is inferred. Complete
original text remains private. Written steps are evidence of what the model
wrote, never proof of hidden reasoning or a latent causal error.

`PER_EXAMPLE_TEXTUAL_GRADIENT_SCHEMA_V2` and
`PER_EXAMPLE_TEXTUAL_GRADIENT_PROMPT_V4` add this projection to one independent
Gradient input per incorrect Optimize example. First-valid Gradient recovery,
the abstraction guard, gradients-only clustering, responsibility and Pattern
selection are unchanged. Clustering receives corrective gradients, not problems,
references or solution text.

`PATTERN_GRADIENT_VISIBLE_SOLUTIONS_MEMORY_INPUT_V5` exposes the corresponding
actual candidate realization in every local panel observation. The next mutation
reads the current local parent's observed solution, final prediction, reference,
correctness and validity. Selected root failure representatives retain their own
provenance and bounded projection. They are not relabeled as child realizations.
The six-generation, 36-metric, six-panel-capacity, four-export, two-Full ceilings,
candidate contract and all selection/transition/stopping rules are unchanged.

Only Optimize trajectories enter adaptive Gradient or Layer1 feedback. Shadow
profiles remain inside the existing private gate. Validation/Test cannot enter
these projections. Solver requests still contain only the immutable interface,
mutable procedure and public problem. Gold/reference answers remain evaluator
and optimizer evidence. Trajectories are not automatically copied into mutable
procedures, clustering inputs, Memory, shared lessons or public reports. Memory
continues its existing structural Situation–Action–Outcome–Lesson projection.

Fresh trajectory-enabled runs require
`MATH_V2_2_VISIBLE_TRAJECTORY_EXECUTION_BINDING_V1`, an exact V6 interface/policy,
new Gradient prompt hash, downstream schemas, fresh source/manifest/startup,
attempt, cache namespace, initial profiling and separate single-use API
authorization. V2–V5 interface bytes, historical schemas, bindings and reports
remain reproducible. Old outputs cannot be retroactively trajectory-enabled.
The current-only builder remains the sole production graph.

Existing frozen Validation accounting lengths may be rebound by the exact
serialized empty-request format delta without reading held-out content; the
new artifact/hash is required. The existing isolated accounting preparation
path also resolves V6. This is request-length metadata, not Validation evaluation
or authorization. The interface correction scope grants no real API access;
the governed prep authorization remains a distinct final requirement.

Offline synthetic conformance must cover both Gradient and subsequent Layer1
mutation, recovery, provenance, held-out rejection and noninterference. A future
separately authorized canary must inspect actual ordinary content, marker
compliance, response length/recovery, explicit thinking=false dispatch, complete
private retention and the two actual feedback inputs. Offline conformance does
not establish qwen3-8b adherence, trajectory usefulness or optimization improvement.
