# All-error same-Parent two-proposal control V2

Independent user-requested control for the completed C0/C1/C2 scheduling study, not a change
to Unified Team Prompt Search or the default multi-Parent kernel. Schema
`single_prompt_full_error_control_draft_v2`; kernel
`SAME_PARENT_TWO_ALL_ERRORS_PROPOSALS_CONTROL_KERNEL_V2`. Manifest:
`experiments/manifests/single_prompt_full_error_control_v2.json`.

## Scientific procedure

Run A2 repeat01/02/03 sequentially, each with algorithm and applicable sampling base seed42
and20 opportunities. Start each repeat from the frozen initial Role/Strategy/Answer Prompt,
with one retained Parent and no population or Memory. At each opportunity deliver **all**
current Optimize150 errors, in original membership order, through the existing `NoModule`
adapter. Each error contains the actual Parent response, reference answer and the native
bounded worked reference view. `structured_feedback` and `module_context` are null. No
Gradient, Cluster, Pattern ranking, representative sampling or additional diagnostic model
is run. The Optimizer receives no Validation/Test outcomes.

Reuse the actual C2 proposal scheduler, streaming transition auditor, `generate_candidate`
and structural contract: one block replacement, three nonempty blocks, no
content/overlap/arbitrary length Guard; initial generation plus at most three structural
retries. A no-op is recorded and skips measurement without forced regeneration. Each
opportunity freezes one current Parent and generates two independently requested candidates
from it. Both receive exactly the same full current error set, without evidence from the
other candidate. Generate both before evaluating either. Do not force different text or strategies; identical changed candidates remain two generation attempts and may share exact Solver observations. Every changed legal candidate
receives full Optimize150 and Validation100. Select the highest Validation correct count,
breaking ties by slot order, and accept at most one iff strictly above the round-start
Parent. No Optimize admission gate or edit merging. A no-op or exhausted slot does not block
the other. A round is generation-exhausted only when both slots exhaust structural retries;
if no candidate is measured and at least one slot is no-op, record a no-op round instead.
Keep the Parent otherwise. No efficacy early stop. The final Prompt is the last retained
Prompt, not the best Test result or a Parent-pool proposal.

The existing C2 same-Parent scheduler and formal candidate generator/evaluator are reused,
not reimplemented. The terminal audit additionally proves that each Optimizer evidence set
equals all current wrong IDs, with actual retained-Parent responses; a three-example subset
cannot silently replace it. Runtime and ledger audits prohibit Gradient/Cluster calls for
this control.

## Matched contracts and comparison limits

Use the scheduling study's original Optimize150, adaptive Validation100 sampled from the
frozen300 with42, formal MATH Test300, initial Prompt, gpt-4o-mini Solver, qwen3.7-flash
Optimizer, SDK2.37.0, decoding/output capacities, pinned extraction and mathematical
scoring. Original dataset partition provenance is preserved, not relabeled seed42. Each
repeat uses the same memberships. Two slots are proposal branches, not a two-Parent pool;
only one running Prompt is retained. The shared artifact field
`diagnosis_source_prompt_sha256` denotes round-start Parent provenance only in this control:
all structured diagnosis is null and all diagnostic counters are zero.

Solver96 for both training and Test. The existing independent16-slot local parse/score
admission queue preserves the original8-second worker deadline. No diagnostic calls are
allowed (the shared transport's unused Gradient capacity is16). Native input/model-context
checks remain technical execution checks; they must not truncate the all-error set or invent
a Prompt character budget.

A2 has20 two-candidate opportunities, at most40 candidates per repeat. C2 also has two slots
when two Patterns exist, but degenerates to one when fewer exist; A2 always has two
independently requested slots, even when the full error set is empty. Do not duplicate or
fabricate Pattern directions. Report the realized candidate/evaluation and Token difference,
not an assertion of identical realized budgets. C0's maximum candidate budget is smaller.
This later control does not retroactively create common Teacher randomness or pair its
Solver realizations with the already completed scheduling study. That study used a32/8
prefix and96/16 suffix; this control starts at96. Seed42 repeats measure provider trajectory
variation, not robustness across distinct seeds. Historical Test exposure is disclosed;
these are exploratory comparisons.

## Fresh execution, cache and resources

No historical checkpoint, final Prompt, Solver cache, paid response, authorization or cost
is imported. Only complete identical Solver terminal requests may be reused within the same
repeat and phase, across rounds. Repeats and phases are isolated. All Optimizer calls are
fresh; no Teacher response cache or replay.

Training successful-semantic plus in-flight ceiling: `3 * (250 * 4 + 20 * 2 * (250 * 4 + 4))
= 123480`. Final Test has a separate `3 * 300 * 4 = 3600` ceiling. These are worst-case
limits, not forecasts. Cumulative accounted Tokens and failed-network physical attempts have
no hard cap. Typed connection/timeout failures use the existing 2/4/8/16/30/60-second
then60-second backoff indefinitely, with each attempt accounted and excluded from
semantic/structural opportunity counts. All other identity, authorization, persistence,
ledger, source, scoring or provider faults fail closed and preserve evidence; no hot repair
or restart is authorized.

Only frozen Optimize/Validation memberships affect optimization. Test content stays sealed
until all three training cells finish and the frozen runner audits all transitions,
evidence, receipts, ledgers, final Prompts and terminal hashes. Then, if the exact new scope
has conditional Test authorization, the same runner derives a fresh Test scope bound to the
training terminal and evaluates all three final Prompts,300 each, Solver only. Test has its
own ledger and empty cache, and rechecks training seals before and after evaluation.

## Preparation and reporting

Preparation is zero-API: `scripts/ablations/prepare.py --full-error-control --destination
<fresh-prep>`. Only `scripts/run_experiment.py --prep <prep> --run-root <train-run>
--execute` may dispatch after a fresh committed source, hash/model/data/budget freeze,
complete offline verification and explicit exact-scope API grant. Existing requests and
historical grants do not authorize an unknown future frozen scope.

Report each repeat's Test/Validation accuracy, updates/rejections/exhaustions/ no-ops,
actual evidence counts, Solver/Optimizer calls and Tokens, semantic recovery, network
conservative charges and cache hits. Report stage and aggregate costs once. Do not infer
diagnosis accuracy or same-family repair transfer from aggregate Test scores. Publish
counters and hashes, never raw private artifacts.
