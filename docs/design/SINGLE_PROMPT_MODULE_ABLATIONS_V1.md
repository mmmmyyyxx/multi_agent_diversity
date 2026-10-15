# Isolated single-Prompt module ablations

New defaults are superseded by [full-diagnosis V5](SINGLE_PROMPT_FULL_DIAGNOSIS_PATTERN_V5.md):
three repeats, all-error C diagnosis before clustering, and formal representative sampling.
The original Batch/five-repeat preparation below describes the historical V1 design.

This user-requested framework is separate from Unified Team Prompt Search and
from every frozen historical single-Prompt attempt. It migrates reusable study
orchestration into `multi_dataset_diverse_rl/research/ablations/`. Historical
controllers, manifests, authorizations, caches, checkpoints and results remain
unchanged. New real attempts require new registration, source/data/phase freeze,
governed handoff and explicit authorization; planning and synthetic smokes grant
none. No real execution binding is issued by this infrastructure task.

## Defaults and repeated-seed identity

Defaults are seed42, five repeats per arm, twenty opportunities per repeat.
Repeat IDs1–5 are part of the cell, request, checkpoint and cache identities.
These are five independent provider realizations with the same algorithm seed,
not five distinct seeds. Seed42 controls error sampling and Pattern ties; the
inherited provider wire does not add a new provider `seed` parameter.
Different repeats must never share response cache entries. Arms within one
repeat share identical Solver requests under the paired realization policy.
Batch selection and Pattern ties retain Seed+round sampling, without a repeat
salt. No new guarantee of different outputs or different strategies is implied.

The default arms are A(all current Optimize errors, no module), B(random three
current errors, no module), and C(the same Batch selection, largest Pattern).
Optimize150, the historical fixed Validation100 subset selected with membership
seed88, and formal Test300 remain preparation defaults. The new experiment seed
does not silently change split membership. Actual dataset/source/membership hashes
must be bound by the future preparation; synthetic fixtures contain no real data.
Training Solver32, Gradient8 and final Test Solver16 retain previous defaults.

## Shared optimizer kernel and module boundaries

All arms use the same independent optimizer instruction, output parser, complete
replacement limits, one-block prompt guards, four total generation draws, candidate
evaluation and strict Validation gain gate. A changed legal candidate is retained
only if it answers at least one more Validation item correctly than the recorded
incumbent. No Optimize acceptance gate is added. Rejected candidates retain both
the incumbent Prompt and its recorded observations. Four invalid, unchanged or
overlong generations consume one explicit exhausted opportunity. Non-generation
failures propagate and do not become rejections or exhaustion.

The common optimizer instruction now supports optional module context for every
arm. This is a fresh common wire identity, not a claim of byte-identical historical
optimizer prompts. Baselines have no implicit optimizer history or LLM Memory.

Each registered module supplies its versioned binding, state capacity, maximum
semantic Teacher calls, initial state, Optimize-only feedback preparation, and
post-opportunity state update. Module objects and persisted JSON states are private
to one arm/repeat. The kernel never sends module state/context to the Solver.
The module interface receives no Validation predictions/scores or Test capability;
only the final candidate decision label crosses the outcome interface. Module
Teacher ports permit Gradient/Cluster calls on Optimize only. Real phase budgets,
reservations, retries and authorization belong to the governed broker, not modules.
Unknown modules fail closed, including unconfigured `memory`.

Checkpoints preserve complete retained observations, candidate evidence, module
state, decisions, costs and identity/integrity hashes. Completed-boundary restoration
checks exact plan/cell/module/data/initial-prompt identities and strict gate history.
It is an offline state capability, not permission to resume a real attempt or import
historical checkpoints. Incomplete paid-response recovery requires its own future
governed binding; this framework does not silently synthesize it.

## Formal modules are imported, not reimplemented

Gradient extraction, input validation, abstraction/content guards, alias transport,
cluster recovery, partition normalization and merging call the actual production
`PerExampleGradientProvider`, `GradientExtractor`, `GradientClusterProvider` and
`score_gradient_partition`. Independent transport adapters initialize the Optimize
evidence contract and override only provider I/O. No Gradient/Pattern algorithm,
prompt template, parser or guard is copied into the study package. Production
dependencies, policy values and prompt hashes enter the module binding/run identity.

The independent C focus is the previously approved largest unique support count,
with canonical Pattern-ID ordering and Seed+round hash-based random ties. The
formal team responsibility selector is not used, and no team responsibility is
invented. Optimizer feedback contains exactly the chosen Pattern and its supported
errors/Gradients/diagnoses; complete diagnostic evidence stays in private audit.
No legal Pattern means the original raw Batch with structured feedback null,
exactly the B feedback projection. Full-support single-Pattern audit compatibility
is retained only when there is no excluded evidence. A future responsibility
ablation must define and authorize its own single-Prompt semantics.

Formal Memory is `PopulationEditMemory`, imported directly; its bounded storage,
retrieval, context budget, transaction and checkpoint algorithms are not replaced.
`production_memory_binding()` describes its current policy and unresolved adapter
questions. `new_production_memory()` constructs the real implementation. It is
deliberately not a registered runnable arm until bootstrap scope, absent team
Vote/Shadow fields, recorded outcomes and optimizer delivery are confirmed. Never
fake a five-member team or reinterpret single-Prompt Accuracy as measured Vote.
Future modules wrap their official implementations behind this same interface and
must declare scientific adapter differences before freeze.

## Offline preparation and verification

`scripts/ablations/study.py --plan` creates an unapproved private draft. `--smoke`
requires the pre-import zero-network guard and exercises the full default15 cells
with fake providers and synthetic Optimize/Validation/Test rows. The only provided
broker is `OfflineAblationBroker`; it cannot authorize or launch real APIs.
`scripts/run_experiment.py --manifest ... --preflight` recognizes the draft schema
but does not make it executable. Runtime artifacts belong below ignored
`runs/ablations/`; share only sanitized reports, metrics and hashes.

Summary output includes per-repeat/group Validation and optional final Test
Accuracy, optimization Token/call totals, update/rejection/exhaustion counters,
and ten within-arm repeat comparisons when five repeats are complete. Paired-cache
physical cost is charged to the first dispatcher, never counted twice. Lexical
similarity does not establish semantic strategy diversity or causal efficacy.

Current team composition/entrypoint eligibility and old frozen studies are unchanged.
