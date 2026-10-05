# Current Research State

The sole active research architecture is Unified Team Prompt Search.

- CURRENT_METHOD: unified_team_prompt_search_v2_1.
- CURRENT_BENCHMARK_SUITE: MATH, IFBench, HotpotQA.
- CURRENT_EXPERIMENT: unified_semantic_contract_v2_1 zero-API implementation audit.
- Execution readiness and authorization come from experiments/current_frontier.yaml.
- Pattern = IMPLEMENTED_DEFAULT_OFF; Memory = IMPLEMENTED_DEFAULT_OFF.
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

## Readiness and next steps

1. MATH: V1.1 reference eligibility, split, initial team, production composition,
   models/provider and finite canary bounds are versioned. Four arms have
   actual public GEPA fake E2E conformance evidence. The authorized A1 Seed81
   first-parent epoch canary aborted during initialization: its first successful
   Solver response lacked the frozen final marker. No search opportunity or
   complete epoch occurred. The authorization is consumed; Formal readiness is
   closed. Private forensic reconstruction proves R2: the system request had
   the output contract, but the response lacked its marker. V1.2 freezes an
   independent immutable formatting interface, effective request identity and
   cache isolation; the strict parser and mutable team remain unchanged.
   Zero-API readiness covers only a fresh unauthorized canary, with real model
   adherence unverified. See the current frontier for exact identities.
2. IFBench: data and new split frozen; pinned checker packages, spaCy model and
   NLTK resources have a verified isolated runtime lock. Aggregation-aware
   responsibility remains RESPONSIBILITY_POLICY_NOT_FROZEN. Ambient runtimes
   must match the lock and explicit NLTK_DATA locator or fail closed.
3. HotpotQA: train/labeled validation and new split frozen; normalized answer
   plurality and binary responsibility retained. SYSTEM_DEPENDENCY_RETRIEVAL_NOT_FROZEN:
   corpus, index, retriever and two-hop summary/query/answer integration are missing.
4. Future cross-benchmark Pattern/Memory factorial: A1 core, A2 Pattern, A3 Memory,
   A4 both. MATH develops mechanisms; IFBench/HotpotQA confirm across tasks.

Global current experiment Solver is qwen3-8b for all members, arms, seeds and
adaptive re-evaluations. Optimizer/reflection and Pattern use qwen3.7-flash.
See experiments/plans/v2_pattern_memory_multibench_v1.yaml; it authorizes no calls.

## Historical preservation

BBH is historical development/replay only. Frozen source, manifests, reports,
V17/V18, Formal V3/V4 and V2 BBH structural tests remain intact. The original
V1 preexecution audit stays PREEXECUTION_BLOCKED. The unexecuted BBH V1.1 draft
is superseded for new work; its creation-time reports remain unchanged.

## Method semantic-contract amendment

V2.1 is implemented with single-mechanism repair, separate coverage metrics,
initial competence floors, strict team deployment gains and grounded experience.
Allocation telemetry is descriptive and observation-only. Historical MATH V2
Canary/Pilot preps remain immutable and unused; they cannot authorize the new
method. Current readiness is HOLD pending a new benchmark execution binding,
source/preexecution freeze and explicit attempt authorization. No real efficacy
was observed for this amendment. Solver/model/data/access contracts are retained.
