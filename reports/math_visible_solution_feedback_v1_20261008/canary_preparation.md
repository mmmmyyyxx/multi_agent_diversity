# Future execution preparation — HOLD

No real execution is authorized by this implementation task. The opt-in profile
is `experiments/execution_bindings/math_visible_solution_offline_profile_v1.json`;
it is intentionally unfrozen and fails before data/provider execution. Historical
Pilot authorization remains closed. No runnable authorized canary is claimed.

The exact implemented feedback contract is defined by
`docs/design/MATH_VISIBLE_SOLUTION_FEEDBACK_V1.md` and the new profile:
V6 final-marker interface, ordinary assistant content, qwen3-8b,
thinking=false, unchanged decoding/recovery/scoring and bounded 4096-character
provenance projection. Gradient input V2/prompt V4 and Layer1 input V5 must be
bound together. Models, original team, memberships, selection, Memory and
transition must not be silently altered in a future profile.

Preparation requires the following sequential steps:

1. Design and register the separately scoped canary; freeze its finite work,
   memberships, roles, initial profile, seed, provider/cost ceilings and access.
   The implemented derivation from the flat current Pilot binding preserves that
   parent's operational bounds. It does not silently create a smaller canary.
   A canary with different bounds needs an explicit new registered binding scope.
2. Materialize the exact V4 Gradient JSON artifact with
   `visible_gradient_prompt_artifact()`, fresh correction scope and cache/attempt,
   and V6 accounting metadata. Use the normal frozen source/hash closure,
   manifest v2, governed preparation and `scripts/run_experiment.py` entrypoint.
   Private draft/synthetic artifacts from this task are never execution authority.
3. Verify model and dispatch fields, full original-response retention, Optimize
   isolation, all frozen dependencies, token ledger and startup identity. Complete
   offline checks and set readiness only after the scientific owner freezes them.
4. Obtain separate explicit single-use API authorization tied to that exact
   source/startup, scope, roles and budget. The correction scope's
   `real_api_authorized=false` is not such authorization.
5. Only then execute the frozen command. Audit ordinary response content,
   unique last final marker, valid final-payload scoring, response lengths and
   recovery counts, selected-member Gradient provenance and current-parent
   Layer1 provenance. Inspect provider behavior directly; never substitute a
   private reasoning field or infer hidden causality.

Publish only hashes, counters, categories and metrics. Inspect private inputs
for provenance and retention without placing problem, answer, prompt or solution
contents into public reports. An operational canary can establish dispatch and
feedback conformance; effectiveness requires its own valid experiment.
