# Single Prompt Gradient–Pattern, five fixed-seed repeats

This fresh independent ablation uses the isolated single-Prompt module kernel.
It does not change the active Unified Team Prompt Search composition or reuse
any historical authorization, checkpoint, paid Teacher response or Solver cache.
The source commit and complete file hashes are frozen in a private preparation.
The machine-readable preregistration is
`experiments/manifests/single_prompt_pattern_repeat_solver_cache_v2.json`.

## Question and contrasts

A optimizes using all current Optimize errors. B optimizes using a deterministic
random Batch of at most three current errors. C diagnoses that same Batch through
the actual production Gradient extraction, clustering and partition validators,
then selects exactly one Pattern with maximum unique supported-error count.
Ties use the existing Seed+round deterministic random selector. Optimizer receives
only that Pattern and its supported errors, Gradients and diagnostics. With no
legal Pattern it receives the original raw Batch and null structured feedback.
The complete Batch diagnosis remains private audit evidence.

C versus B measures the extra diagnosis and selected feedback pipeline; its
additional Teacher cost is reported separately. A versus B measures feedback
volume. These contrasts do not isolate Gradient from Pattern individually.
The registered Gradient-only extension is not an arm of this experiment.

## Fixed procedure

Each arm has five repeats, all with algorithm seed42 and separate repeat IDs1–5.
Order is A1–A5, B1–B5, C1–C5, sequential. Error sampling and Pattern ties use
Seed+round without repeat salt. The provider wire adds no sampling seed. These
are stochastic repeats with a fixed algorithm seed, not five distinct seeds.
All start from the same frozen structured Role/Strategy/Answer Prompt.

Each cell has twenty opportunities. One legal, changed, single-block candidate
is generated per opportunity, with the initial draw plus at most three retries
for malformed, overlong or unchanged output. Four unsuccessful draws explicitly
exhaust one opportunity. Every legal candidate is scored on Optimize150 and the
same frozen Validation100 subset selected from Validation300 with membership
seed88. It is retained only with at least one additional correct Validation
answer over the recorded incumbent. Optimize accuracy imposes no acceptance
threshold. Incumbents and their recorded scores are not resampled after rejection.
No efficacy-based early stop, implicit history or Memory is enabled.

## Solver-only cache amendment

Only terminal scored Solver observations may be shared within the same repeat
ID, across arms and rounds. Identity includes phase/split, seed, repeat, full
structured Prompt, problem, stable example identity, scoring reference, model,
decoding parameters and complete native request. Invalid first responses are
not cached as successful observations; the complete bounded semantic recovery
result is cached. Different repeats and different splits/phases remain isolated.
No historical cache is imported. Reflection/Optimizer, Gradient and Cluster
always dispatch fresh requests, including byte-identical messages; paid response
replay is prohibited. This is V2 identity, incompatible with V1 checkpoints.

Training Solver concurrency32 and Gradient8; final Test Solver16. Solver/model,
SDK2.37.0, decoding, four semantic draws, capacities3600/6144, scoring and gold-blind
answer extraction match the existing MATH contracts. Teacher uses the frozen
native nonthinking generation contracts. APITimeoutError/APIConnectionError are
accounted and retried after2/4/8/16/30/60 seconds, then60 seconds indefinitely
until success or human stop. They consume no semantic draw, format retry or
optimization opportunity. Other failures abort and preserve evidence.

Training successful semantic responses plus in-flight attempts are bounded by
317400; Test separately by18000. Network-failed physical attempts are excluded
from those bounds and have no finite retry limit. Total Tokens have no hard cap.
Every physical attempt has a reservation, receipt and charge; unreliable usage
is conservatively bounded, never described as supplier-reported consumption.

## Phase governance and analysis

Fresh exact user authorization is required for training plus adaptive Validation;
conditional final Test authorization is separately explicit in the same scope.
Training cannot read Test content. All fifteen completed cells, twenty contiguous
opportunities each, strict gate transitions, final Prompts, original source/data
identities, receipts and accounting must pass checks before final Test opens.
Test evaluates all fifteen final Prompts on the formal-aligned MATH Test300,
with its own ledger and repeat-scoped Solver cache. No Test outcome changes the
method or selects another Prompt during execution. Test has already been observed
in previous studies; this experiment is not a new independent confirmation.

Report per-repeat and group Test/Validation accuracy, updates/rejections/exhaustion,
training and Test calls/Tokens by role, cache access and reuse, and lexical Prompt
similarity and correctness disagreement. Shared Solver costs are charged once
to the first dispatching cell, so arm totals are order-dependent; report study
totals and role counts as well. Five repeats are paired within repeat ID, with
no claim of five independent-seed trials or semantic diversity from lexical
distance alone. All raw Prompts, questions, answers and responses stay private.
