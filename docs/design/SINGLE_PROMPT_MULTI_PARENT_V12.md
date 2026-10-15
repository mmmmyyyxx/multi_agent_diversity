# Single-Prompt Gradient–Pattern with native Parent policies V12

User-confirmed independent adaptation, not a change to the five-member team method.
The new preparation default is `current_config()` / `pattern/default.json`, schema V6,
kernel `SINGLE_PROMPT_MULTI_PARENT_KERNEL_V6`. The explicitly named V5 factory
`default_config()` remains a predecessor API for existing offline tests; it is not
the new default configuration artifact. No old candidate content Guard is restored.
New real attempts require a fresh registered, frozen manifest, source, authorization
and handoff. This implementation and its draft do not authorize an experiment.

## Shared A/B/C search kernel

Each arm/repeat has one running Prompt and one separate pool of at most three
canonical Prompt hashes. The pool bootstraps from the initial complete Optimize
profile. Its fitness is correct/N on the fixed complete Optimize150 membership.
There are no dummy peers, Team Vote values, Shadow scores or Group insights.
Measured profiles, source Prompt hashes and membership provenance remain private.

Directly instantiate the production `FitnessWeightedSampling`, `BestThreeRetention`
and `PerMemberArgmaxAssembly`. Do not copy their algorithms. The single-Prompt
container adapts the actual one-lane measured accuracy interface; the production
five-member `Fitness`/`ParentPopulation` containers cannot represent one Prompt
without fabricated team context and are not used. Bind the native policy source
hash in the new identity. The production team graph and its policies are unchanged.

Before diagnosis or generation, freeze the current pool and sample three Parents
with replacement. Use the production epsilon-softmax probabilities, beta4 and
uniform mixture0.1, with canonical hash tie order. Every Parent has positive mass.
Draw all three first; no Child generated in this opportunity becomes its Parent.
All three slots independently generate one block replacement. Preserve the
independent kernel's initial attempt plus three format retries per slot, rather
than silently changing that limit to the team's two draws. A no-op ends its slot
without evaluation or regeneration; malformed exhaustion abandons that slot.
The kernel always completes20 opportunities, not20 candidate calls.

Diagnose the running Root once per round. A uses all Root errors; B uses seeded
up-to-three Root errors; C uses all Root errors for the actual formal per-example
Gradient and success-only clustering/partition validation, then the approved
maximum-support Pattern and formal up-to-three representative sampling. No legal
Pattern retains the approved raw-B fallback. The three slots share this direction.
Each Optimizer also receives its selected Parent's actual cached Optimize response
on the representative errors, explicitly labeled separately from Root diagnoses.
Root errors need not be errors of another Parent. This evidence never creates an
additional Solver input channel. Reference projection and readable observation
helpers are actual production functions. Group and independent Memory remain off.

Every changed legal Child obtains a complete Optimize profile and is upserted into
the pool immediately, independently of Validation. Retain the three highest current
fitness entries; tied hashes sort ascending. No initial/running Parent is protected.
Identical Solver requests can reuse their exact terminal observation; Teacher calls
are fresh in every slot/round/repeat. Pool state never crosses cells/repeats.

Among this round's changed Children choose highest Optimize count, then ascending
canonical Prompt hash, then slot number. Only this winner reaches Validation100.
Update the running Root only if it answers at least one more Validation question
correctly. There is no added Optimize competence/progress gate. Other Children
receive no Validation measurement. Record/reuse each already measured Validation
profile; do not reroll an incumbent or a previously evaluated identical Prompt.

At closure, the native argmax policy selects exactly one pool proposal before any
new Validation access. Its Validation must be no lower than the last running Root's
recorded profile. Neutral final admission is allowed; per-round updates still
require strict gain. Reuse a recorded proposal profile when present. Final rejection
is explicitly `FINAL_PROPOSAL_REJECTED`; `final_prompt` continues to identify that
proposal and the safe running Prompt has its own field. No silent fallback,
Validation ranking of the pool, reselection or retry. A rejected proposal cannot
open the automatic Test phase. A partial Test scope needs separate future governance.

## Seeds, data and accounting

All new sampling base seeds use42, including adaptive Validation subset selection.
Native Parent RNG scope binds the scientific policy and Optimize membership;
round identity and actual ordered pool determine its draws. Do not salt with arm,
repeat number, cache namespace or run path. Same states at the same round produce
the same draws across repeats; different measured pools may evolve differently.
The existing generation contracts expose no provider generation-seed binding;
do not claim that LLM realizations are deterministic from the algorithm seed.

No original MATH Optimize150 / Validation300 / Test300 partition is resampled here.
New Validation100 is drawn from that fixed Validation300 with42 and gets a new
membership identity. Historical subsets sampled with88 are preserved and never
relabeled. Solver caches remain exact-request, same-repeat, same-stage only,
shared across A/B/C and rounds; no old-run cache or checkpoint is imported.

For nine default cells and20 rounds, the conservative training successful-semantic
and in-flight ceiling is437940, including initial Optimize/Validation, up to three
complete Optimize Children per round, one Validation winner, closure Validation,
three four-draw Reflection slots, and the full C diagnosis ceiling. Final Test
has its independent10800 ceiling. Exact request reuse may lower actual calls;
charge shared Solver cost to its first dispatcher. Tokens and network-failure
attempts remain uncapped under the existing accounted recovery policy.

Whole checkpoints bind running Root, recorded profiles, complete pool, frozen
Parent selection, all slot outcomes, module state and strict-gate transitions.
An offline audit reconstructs native sampling/retention, checks slot receipts and
evidence, validates pool state separately from Root updates and verifies the final
argmax/nonregression decision before Test. No partial checkpoint continuation is
silently enabled: a future resume contract must bind this entire state and costs.
Reports distinguish opportunity updates/rejections/exhaustion/no-op from slot
counts, measured Children and final admission. These are new-protocol results;
historical single-Parent A/B/C accuracies are not matched causal controls.
