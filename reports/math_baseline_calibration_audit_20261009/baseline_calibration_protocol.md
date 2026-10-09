# MATH Baseline Paired Calibration Protocol V1

DESIGN ONLY. No real API authorization, no execution handoff and no paid runner.
This is an independent baseline diagnostic, not the V2.5 optimization treatment.
The dated directory identifies the requested task; the audit was completed on
2026-10-10. The machine-readable freeze is
`experiments/protocols/math_baseline_calibration_v1/protocol.json`.

## Arms and causal comparisons

All arms share Role and Strategy:

```text
You are a helpful assistant.
```

```text
Answer the question.
```

Only the following planned Answer instructions are published. These are the
user-specified calibration protocol, not excerpts from private responses.
The renderer remains `SystemPrompt` with Role, Strategy and Answer blocks.
The User message is the exact original problem, with no optimizer context.

Arm A:

```text
On the last line, write "Final answer:" followed by your mathematical answer.
```

Arm B:

```text
End your response with a final answer on a separate line.
The line must start with the exact text "Final answer: "
followed by only the mathematical result.
Do not use a Markdown heading, bullet, or bold formatting
for this line. Do not write anything after it.
```

A and B both use the unchanged
`MATH_EXPLICIT_FINAL_ANSWER_EXTRACTION_V2` native Parser. A/B is the
Prompt-only contrast, subject to stochastic provider realization uncertainty.

Arm C:

```text
On the last line, write only your final mathematical answer
in the form \boxed{...}. Do not write anything after it.
```

C uses `MATH_BASELINE_BOXED_TERMINAL_V1`: only one complete balanced box on
the last nonempty line, optionally enclosed in outer math delimiters, with no
prose, trailing punctuation or suffix content. The payload must pass the same
pinned mathematical parser. Truncated and non-stop responses remain invalid.
Intermediate boxes never determine the terminal result. This is an interface
inspired by boxed mathematical outputs, not a replication of C-EVOLVE.
A/C and B/C compare complete output interfaces, including different Parsers.

## Fixed model, sampling and data

All arms use `qwen3-8b`, `enable_thinking=false`, temperature 0.2, top_p 0.8,
top_k 20, min_p 0, presence_penalty 0, frequency_penalty 0 and initial output
capacity 3600. Model and reasoning mode are not experimental factors.
Provider alias alone cannot attest backend weights; no new identity probe is
authorized. Any provider/model identity failure requires a factual abort.

Use only the existing frozen Optimize150 membership. Exclude the 12 V2.5
Canary example IDs before sampling: the remaining legal pool is 138. Select
60 by subject/level largest-remainder allocation with lexical ties, then
SHA256 of protocol identity, seed 81 and stable example ID within each stratum.
Preserve original frozen Optimize order. Outcomes never enter selection.
Every reference must pass the pinned scorer's reference-validity check before
the future run can be frozen. If fewer than 60 legal examples are available,
freeze the actual count and recalculate budgets; never open another split.
The JSON freeze records subject/level coverage, uncovered strata and only
hashed example identities. Exact IDs, problems and references stay private.

The Optimize pool has already served adaptive development. Excluding Canary12
does not make the remaining examples an untouched evaluation set. No access
to Shadow, Validation or Test is included, and no generalization claim is made.

## Stages, independence and execution order

Stage 1: one Solver realization per example per arm, 60 identical examples,
180 logical evaluations. Within-arm example order is identical. Across arms,
cycle ABC, BCA, CAB by example index, solver concurrency 1. Record actual
dispatch order and failures. All optimizer components remain disabled.

Stage 2 is design only: five independent member realizations for every arm and
example, the same 60 examples, 900 logical evaluations. A separate review of
Stage 1 and fresh exact authorization are required. Use benchmark-selected
equal-equivalence plurality consistency and record Member, Vote and Oracle.
Each member, arm and stage has a distinct realization/cache/accounting identity;
no output copying and no Stage 1 response reuse. The fake implementation only
verifies identity lanes; it does not execute or claim a real team evaluation.

## First draw, recovery and cost

First draw means the first successful semantic generation, excluding transport
errors. It is the primary Prompt-following and mathematical ability comparison.
Recovered metrics use at most four successful semantic generations, triggered
only by native output invalidity. A valid but wrong answer stops immediately.
Transport has at most 21 physical attempts per draw; it never consumes a
semantic draw. Operational failure does not create a completed logical score.

Capacity is 6144 only when the immediately previous successful semantic response
was truncated; otherwise it is 3600. Prompts and all sampling fields remain
identical through recovery. This is the existing rule, not a permanent expansion.
Record each draw, physical retry, input/output usage, conservative unknown-usage
charge, capacity trigger, finish reason, native reason, terminal result and
tokens per logical example. Keep first-draw denominator N and later conditional
eligible denominators distinct. Cost per correct is undefined when none are
correct, not zero. Monetary price is unknown without authorized billing data.

Stage-specific proposed charged-plus-reserved limits are in
`baseline_calibration_budget.json`. Their historical-rate scenarios do not
forecast effects of B/C or the new 60 examples. Each cap is independent and
may terminate a stage before a complete panel. Exhaustion publishes partial
operational counters; no incomplete-arm efficacy comparison or adaptive
reallocation. The V2.5 consumed 2M scope cannot fund either stage.

## Scoring and paired interpretation

Primary is each arm's native extraction followed by the same pinned mathematical
equivalence scorer. Secondary is `MATH_BASELINE_COMMON_DIAGNOSTIC_V1`, applied
identically to all raw outputs, with extraction completed before seeing Gold.
It accepts explicit plain/decorated labels, an immediately following complete
math/display expression, explicit natural-language final assertions, terminal
whole boxed expressions and unmarked whole mathematical terminal lines.
It rejects inconsistent declarations and unparseable candidates, does not search
arbitrary intermediate expressions and records unknown when no explicit result
exists. Truncated candidates are marked separately. Secondary never changes
Primary, native retries, team votes, selection, prompts or optimizer memory.

Pair all comparisons on exact example ID. Report first-draw format validity and
native correctness; recovered validity and correctness; average tokens and
cost per correct; diagnostic status/correctness; truncation, category and
subject/level breakdowns. Report improved AND regressed example counts for
A/B, A/C and B/C. Stage 2 pairs example-level Member/Vote/Oracle outcomes,
never treats five correlated members as five independent examples.
Optional paired-example bootstrap uses seed 81 and 10,000 resamples, with
percentile 95% intervals. It is descriptive development uncertainty, not proof
of population significance. Exact IDs and per-example outcomes stay private.

Interpretation records four possibilities without a minimum-accuracy gate:

| Outcome pattern | Interpretation |
|---|---|
| Native format improves, common diagnostic mathematics shows no clear gain | Format-only improvement supported within the diagnostic panel |
| Correctness improves while output validity is stable | Math correctness improvement consistent with the panel |
| Validity and common diagnostic correctness both improve | Mixed improvement; retain paired details and uncertainty |
| Incomplete panel, material paired regressions or unresolved uncertainty | Inconclusive; publish limitations without opportunistic reruns |

For C, improved native correctness alone remains an output-interface result.
No outcome revises the old Canary or automatically changes production Seed,
Parser or the V2.5 method. A later change would need a separate scientific
amendment, identity and meaningful tests. Valid unfavorable results do not
justify a rerun. Stage 2 review concerns identifiable interface behavior,
cost and evidence integrity, not an artificial optimization qualification.

## Exact authorization required before any paid run

Freeze a new source commit, this protocol and membership hashes, the exact
stage/roles/model/provider policy, seed/order/concurrency, native and diagnostic
identities, retry/capacity limits, fresh cache and ledger scope, charged plus
reserved cap, stop conditions, split policy and single-use attempt ID. Verify
an explicit user authorization covering those values and the governed handoff.
`READY_TO_RUN=false` and `real_api_authorized=false` throughout this task.
The available CLI supports offline audit, dry run and closed synthetic fake
fixtures only. A production-capable calibration runner does not exist here.
