# Current Research State

The sole active research architecture is Unified Team Prompt Search.

- CURRENT_METHOD: unified_team_prompt_search_v2_3.
- CURRENT_BENCHMARK_SUITE: MATH, IFBench, HotpotQA.
- CURRENT_EXPERIMENT: a4_v23_only_pilot_v1, preparation pending exact API approval.
- Execution readiness and authorization come from experiments/current_frontier.yaml.
- CURRENT_RUNTIME: one complete Gradient Pattern / Rolling Memory / bounded Layer1 bundle.
- Older Pattern/Memory/search combinations: EXPLICIT_HISTORICAL_REPLAY_ONLY.
- SCIENTIFIC_EFFICACY_VERIFIED = NO; SOTA_VERIFIED = NO.

The current explicit opt-in [Pattern contract](../design/PATTERN_GRADIENT_DISCOVERY_V4.md)
uses per-example textual gradients, gradients-only clustering and unchanged support
Responsibility. WHO and bounded rolling Memory retain their existing rules. Code
conformance has no efficacy or latent-cause interpretation. Execution status and
the need for fresh freeze/authorization remain governed by the frontier.

## Dataset and experiment split layers

Canonical datasets are frozen independently of experiment-facing memberships.
The new benchmark_experiment_split_v1 has Optimize150, Shadow300, Validation300,
Test300 for MATH/HotpotQA and Test294 for IFBench. IFBench canonical manifest and
old Optimize/Shadow/Test memberships remain unchanged; Validation is unused train.
MATH V1.1 filters reference-invalid rows before the same subject-proportional
deterministic split algorithm. Canonical source bytes and prior splits remain
unchanged. The frozen reference-validity policy is independent of efficacy.
HotpotQA project Test comes from official labeled validation, not official test.
All splits passed ID/content/input isolation before any model efficacy run.

Optimize supplies adaptive search. Shadow supplies the winner-only adaptive gate.
Validation is post-freeze development evaluation with no online feedback. Test
is sealed. Data availability does not confer held-out model access.

## V2.3-only execution scope

V2.3 is the sole active scientific method; V2.2 and paired execution are permanently retired.
The production entrypoint is `scripts/run_experiment.py`, using `MATHEvidenceBinding`,
the complete current policy bundle and `build_current_team_prompt_search`.
Missing or historical method, evidence and trajectory identities fail closed.

Five qwen3-8b members use V6 ordinary visible mathematical steps and one final
answer line, with thinking disabled. qwen3.7-flash supplies independent
trajectory/reference Gradients, gradients-only clustering and conservative
Layer1 edits. Equal-weight equivalence plurality and fixed peers remain.
Mutation, SearchValidation and TeamProbe are disjoint Optimize memberships.
Layer1 keeps six generations, 42 metric calls and four exports; at most two
candidates reach Full. The immutable initial competence floor, nonregressing
Full Vote, strict target OR team progress and winner-only Shadow gate remain.
Measured initial competence and edit effects enter bounded Memory; uncommitted
candidate coverage cannot replace committed competence. Team-epoch no-commit
patience remains two. Operational horizon/budget stops do not prove saturation.

Real-model adherence and efficacy remain unverified. Fresh source, manifest,
request, attempt, cache and ledger identities precede exact single-use API
approval. Historical authorizations grant no new access. Validation is not
authorized and Test remains sealed. Current execution uses MATH; IFBench and
HotpotQA contracts remain maintained, with their execution blockers intact.
