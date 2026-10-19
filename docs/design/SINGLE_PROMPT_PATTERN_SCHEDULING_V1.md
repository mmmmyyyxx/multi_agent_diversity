# Gradient–Pattern multi-direction single-Parent scheduling V1

Explicit independent study, not a production Team-aware change or the V12 Parent
population. Audit base `ccd97d81825271ee8a5ff9a12a54ad371a77e718` equals the user's
reference SHA. This specification supersedes the earlier **unfrozen** same-Pattern
two-draw draft from this task. No historical manifest, source snapshot, result or
authorization is amended. Real execution requires its own later source freeze.

## Scope and controlled groups

| Arm | Pattern policy | Max candidates / round | Max accepted edits / round |
|---|---|---:|---:|
| C0 Single Largest | Largest unique support | 1 | 1 |
| C1 Sequential Cumulative | Frozen Top-2, latest retained Parent at each step | 2 | 2 |
| C2 Parallel Competition | Frozen Top-2, same round-start Parent | 2 | 1 |

Schema `single_prompt_pattern_scheduling_draft_v1`; kernel
`SINGLE_PARENT_PATTERN_DIRECTION_SCHEDULING_KERNEL_V1`. New identity is opt-in.
Existing single-Parent C0 factories, V12 defaults and team runtime stay unchanged.
New C0 is matched to C1/C2's current structural Guard, models, evidence and scoring;
old C under another Guard/source is descriptive only and cannot replace it.

Each arm has20 outer opportunities and three repeat IDs,all applicable algorithm
and sampling base seeds42 without repeat or arm salts. There is no efficacy early
stop,Memory,WHO,Vote,Probe,Population,Pareto,merge,intervention,hint,pseudo-label,
extra feedback model or learned Pattern weight. Fixed-seed repeats mainly observe
provider/Optimizer randomness; no supported provider generation seed is invented.

## Shared diagnosis and evidence

Use the actual `FullDiagnosisPatternModule`: obtain the current round-start P0's
complete Optimize150 wrong set, extract formal per-example Gradient, then cluster
only successful evidence with actual partition normalization/validation. One
diagnosis per outer opportunity, never another after an accepted first step.
Formal bounded per-item/cluster recovery remains unchanged. Complete diagnostics
are audit-only, not accidentally delivered to the Optimizer.

Top-1 is the existing `LargestUniqueSupportSelector`. Select Top-2 by calling the
same selector on the remaining distinct IDs, preserving its Seed+round tie rule.
C1/C2 use identical ordered foci for identical diagnosis inputs. They can diverge
in later rounds because their retained Prompts and actual diagnoses diverge.
Teacher responses are not shared across arms: even identical initial Parents may
produce different realized diagnoses. Matching the ranking policy does not force
identical realized Patterns or constitute common-random-number Teacher control.
No fabricated team responsibility labels or LLM Pattern ranking exists.

Use actual `project_full` and `sample_pattern_support`,up to three representatives
per focus. Only one selected Pattern,its true original support membership and
representative raw errors/Gradient/diagnoses leave the projection. Unselected
Patterns,nonrepresentative examples and partition audit never enter Optimizer.

No Pattern: one native raw up-to-three B-style Batch,structured feedback null.
One Pattern: one proposal for all arms. Two or more: C1/C2 select exactly two;
never duplicate one direction or switch to a third after a no-op/failure. Separate
selected,attempted,evaluated distinct focus counts and fallback/skip statuses.

## Sequential C1

Freeze P0,G1,G2 before generating A. Generate and fully evaluate A,accept iff its
Validation correct count is greater than P0. Finish that decision before B starts.
B edits the actual retained A or P0, and is compared with **that Parent's recorded
Validation**,not the round-start baseline. For example,if A improves and B falls
below A but stays above P0,B is rejected. Two accepted steps accumulate in order.
This is not a two-candidate tournament.

G2 remains the original P0 diagnosis with the original ID and support set. Use
the already measured current Optimize profile to intersect that support with
still-wrong examples. If empty,record `SKIPPED_NO_CURRENT_ERRORS`,zero generation
draws and no replacement direction. If P0 was updated,call the same formal sampler
on the still-wrong support subset using the frozen P0 sampling state/seed/round.
Restore original support membership in the delivered Pattern and record the
active sampling subset separately. No re-extraction,re-clustering or new call is
hidden in this projection.

When diagnostic source differs from the actual Parent,retain P0's raw errors and
Gradient as historical evidence and explicitly add `diagnosis_context`: source
Prompt hash,current Parent hash,original/active support IDs and the current
Parent's actual readable Optimize responses on these representative examples.
Use native `adaptive_trajectory` and `readable_observation`; no Validation score
or outcome,other candidate content or other Pattern is delivered. The appended
provenance instruction distinguishes old diagnoses from current observations.

Every candidate edits one block. Record its immediate Parent and block lineage,
whether it overwrites the first accepted edit's block,and how many of the first
step's repaired original errors are preserved or broken by B. Acceptance does
not assert semantic preservation of the first repair principle or zero collateral
damage. These are measured diagnostics,not extra gates.

## Parallel C2 and baseline C0

Freeze one P0 and feedback per slot. Generate both independent candidates before
any candidate evaluation. Each Optimizer sees only its own focus. Explicit V1
transport scheduling is serial; scientific parallelism means independent same-
Parent proposals,not simultaneous provider calls. No shared mutable Broker race
or candidate-to-candidate feedback is introduced.

Fully evaluate every changed legal candidate on Optimize150 and Validation100.
Choose highest Validation correct count; ties use **frozen Pattern order / slot**,
not Optimize score,Prompt hash or another performance signal. Accept at most one
iff strictly above P0. Never combine their edits. C0 has one slot with the same
unchanged full-Pattern,projection,generation,evaluation and strict-gain algorithm.

All arms use current Role/Strategy/Answer nonempty structure,single-block editing
and native relaxed unambiguous parsing. User explicitly reconfirmed no keyword,
overlap,answer-copying or arbitrary character-limit Guard. No-op skips without
regeneration; structural failure has initial generation plus three retries.
One failed slot does not prevent another legitimate slot. Non-network integrity
faults fail closed; typed network failures keep existing accounted recovery.
No Optimize/minibatch or team performance gate is added.

## AgentGrad source distinction

Reference: [AgentGrad v2,Sections4.3–4.4 and Algorithm1](https://arxiv.org/html/2609.08572v2),
accessed2026-10-19. Its aggregation forms semantic minibatches/generalized
gradients. Updates follow descending minibatch size and accepted changes become
the next current Prompt; acceptance checks minibatch improvement before Validation.
Its upstream supervision uses intervention-corrected agent outputs.

This study uses repository Gradient/Cluster and reference evidence,no intervention
or new supervision,only Top-2 directions,full Optimize measurement and Validation-
only admission. C1 is **AgentGrad-inspired**,not Original AgentGrad. The paper's
cluster-size schedule and MAS target identification are not silently imported.
Paper results are not evidence that this simplified scheduler works here.

## Data,resources and integrity

Keep frozen Optimize150 and formal MATH Test300,initial Prompt,gpt-4o-mini Solver,
qwen3.7-flash Teacher,SDK2.37.0,current score/decoding/output contracts. Per explicit
user clarification,new adaptive Validation100 is sampled from frozen300 with42,
shared by every arm/repeat. Historical seed88 subsets and original benchmark
partition provenance remain unchanged. The separate six-case reasoning audit
does not remove members or rewrite scores in this experiment.

Only exact Solver terminal observations share cache within one repeat and phase
across arms/rounds/slots. Different repeats/phases never share. Teacher always
calls anew; no old checkpoint,cache,response,authorization or costs imported.
Training Solver32,Gradient8,TestSolver16 remain. Identical candidates may share
one Solver realization,with payment attributed to the first dispatcher.

180 outer opportunities; at most300 candidates. Per cell ceiling:
`250*4 + 20*(max_slots*(250*4+4)+453)`;453=150*3 Gradient+3 Cluster **once per round**.
C0 training90420,C1 training150660,C2 training150660,total391740 successful-semantic
plus in-flight calls. Final Test independently10800. These are worst-case ceilings,
not expected bills. Missing directions,format failures,no-op and resolved targets
reduce actual evaluations. Tokens/network physical failures have no hard cap;
typed timeout/connection errors retain2/4/8/16/30/60 then60-second indefinite
backoff,each attempt recorded and excluded from finite semantic counts.

Each slot has unique stage/proposal identity and private artifacts,full measured
profiles and actual immediate Parent,root diagnostic source,block lineage,gate
baseline,decision,repair/regression breakdown and generation/evaluation costs.
Persist every legal rejected candidate,not just winners. Round records include
shared diagnosis cost,selected/attempted/evaluated directions,accepted step count,
parallel winner and actual final retained state. Record search wall-clock time
separately from preflight,terminal audit and Test.

Stream adjacent sealed checkpoints before Test; reconstruct direction projection,
stale-diagnosis/current-observation mapping,each sequential transition or parallel
winner,all generation artifacts,receipts and complete role/cell ledgers. Boundary
restoration validates prefix seals,semantics and inherited usage before skipping
completed Teacher calls. Any paid partial next round is rejected before new
generation; no silent replay/regeneration or double charge. Real runner still
requires a fresh frozen authorized continuation to resume an existing attempt;
the kernel boundary interface is not permission to reuse a consumed scope.

## Analysis and execution eligibility

Report fixed20-opportunity outcomes and cost-ordered retained-state curves at
common candidate evaluation budgets. C1 has a recorded state after each substep;
C2's winner is available only after all evaluations,so do not backfill its future
winner at a one-candidate budget. Compare last available boundary at or below a
budget,without interpolation or unmeasured intermediate Test accuracy. Report
actual candidate counts,logical Solver metric observations,physical calls,role
Tokens and wall time; equal maximum candidates do not imply equal cost or realized
opportunities. Shared-cache dispatch order affects physical cost allocation.

Distinguish accepted **edits** from updated outer rounds,C1 double-accepts and
resolved-direction skips. For C2,report distinct second-Pattern wins and full
same-Parent correctness disagreement/first-only/second-only repair,including
nonwinning candidates. These scored changes do not establish diagnosis truth or
same-error-family generalization; see the
[historical validity audit](../../reports/pattern_diagnosis_transfer_audit_20261019/README.md).

Opt-in configuration: `experiments/ablations/pattern_directions/default.json`.
Plan/synthetic smoke: `scripts/ablations/study.py --config ... --plan/--smoke`.
Fresh freeze: `scripts/ablations/prepare.py --pattern-directions --destination ...`.
Only `scripts/run_experiment.py --prep ... --run-root ... --execute` can enter a
new exact governed attempt after clean committed source,full offline checks,
source/data/models/budget/cache/retry/access freeze and explicit human API grant.
Final Test is sealed until all nine training cells and their terminal ledgers/
receipts pass checks and separate Test access is authorized. No real API,
historical resume,Test access or Git push is authorized by this implementation.
