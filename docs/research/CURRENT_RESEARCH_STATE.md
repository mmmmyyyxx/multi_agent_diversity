# Current Research State

The sole active research architecture is Unified Team Prompt Search.

- CURRENT_METHOD: unified_team_prompt_search_v2.
- CURRENT_BENCHMARK_SUITE: MATH, IFBench, HotpotQA.
- CURRENT_EXPERIMENT: NO_AUTHORIZED_REAL_EXPERIMENT.
- REAL_EXECUTION_READY = NO; REAL_API_AUTHORIZED = NO.
- Pattern = IMPLEMENTED_DEFAULT_OFF; Memory = IMPLEMENTED_DEFAULT_OFF.
- SCIENTIFIC_EFFICACY_VERIFIED = NO; SOTA_VERIFIED = NO.

## Dataset and experiment split layers

Canonical datasets are frozen independently of experiment-facing memberships.
The new benchmark_experiment_split_v1 has Optimize150, Shadow300, Validation300,
Test300 for MATH/HotpotQA and Test294 for IFBench. IFBench canonical manifest and
old Optimize/Shadow/Test memberships remain unchanged; Validation is unused train.
MATH uses original subject proportions and deterministic largest remainder.
HotpotQA project Test comes from official labeled validation, not official test.
All splits passed ID/content/input isolation before any model efficacy run.

Optimize supplies adaptive search. Shadow supplies the winner-only adaptive gate.
Validation is post-freeze development evaluation with no online feedback. Test
is sealed. Data availability does not confer held-out model access.

## Readiness and next steps

1. MATH: canonical data and new split frozen; equivalence evaluator and binary
   responsibility frozen. Next freeze task-valid initial prompts, V2 production
   composition, provider/resource governance and a real preexecution/canary.
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
