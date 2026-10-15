# Single-Prompt Pattern C-only rerun V7

The user requested rerunning C after the formal-flow audit. This fresh subset
uses the unchanged [V6 kernel and interfaces](SINGLE_PROMPT_MODULE_ALIGNMENT_V6.md),
not the stopped Batch-based C implementation. Only C repeats1/2/3 run, sequentially,
all algorithm Seed42, each twenty opportunities. A/B, repeat4/5 and other seeds
must not dispatch. All three start from the frozen original Prompt. No old state,
checkpoints, responses, Solver observations, costs or authorizations are imported.

Every opportunity diagnoses all current Optimize errors through the actual formal
per-example Gradient extractor, then clusters only successful gradients through
the actual formal Cluster/partition validation. Select one Pattern by maximum
unique support, using the approved Seed-plus-round tie rule. Use the actual formal
representative sampler for up to three supports; only those examples, Gradients
and diagnostics plus the selected Pattern enter the Optimizer. A missing legal
Pattern uses the original seeded B raw Batch fallback, without structured feedback.
Gradient/Cluster known generated-output failures, including ordinary truncation,
receive three total draws. Exhausted per-example Gradient evidence is excluded
from clustering; Cluster exhaustion uses the same B fallback. Integrity, source,
thinking contradictions and unknown failures remain fatal.

Initial generation plus three retries apply to malformed, overlong or unchanged
candidate output. Exhaustion consumes one opportunity; legal candidate evaluation
may be rejected without strict Validation100 improvement. Retention requires at
least one more correct answer than the recorded retained Prompt, which is not
remeasured. There is no efficacy early stopping and no Optimize acceptance gate.

All unchanged roles, SDK2.37.0, models gpt-4o-mini/qwen3.7-flash, output/context
capacities, memberships, score/decoding contracts and concurrency remain V6:
training Solver32/Gradient8 and final Test Solver16. Solver terminal observation
cache is limited to identical complete requests within one repeat and phase;
different repeats/phases and old experiments are isolated. With only C, the
cross-arm part of the same cache rule has no eligible A/B consumer. Gradient,
Cluster and Optimizer never reuse responses.

The derived training successful-semantic plus in-flight ceiling is90420 and final
Test ceiling3600. Failed network physical attempts and cumulative Tokens have no
hard cap. Accounted APITimeoutError/APIConnectionError back off2/4/8/16/30/60
seconds then60 seconds indefinitely, without consuming generated-output draws or
opportunities. Each failed attempt retains its accounting and receipt evidence.

Training never opens Test. After all three complete, source/authorization,
checkpoints, strict gate transitions, receipts, per-role/cell accounting, final
Prompts and terminal seal must pass the unchanged audit. Conditional authorization
may then allow all three final Prompts on the same formally aligned MATH Test300,
in an isolated Test cache and ledger. Test never chooses a Prompt or changes C.

Historical A/B results remain external descriptive references, not newly matched
V6 controls: V6 also corrected shared Optimizer evidence/guards. No claim of an
isolated causal C-versus-old-A/B effect follows from this rerun. Historical Test
exposure is disclosed; the new Test is not called an independent confirmation.
Old C results are excluded from the new C summary. Costs report this attempt only;
history is not silently added or called free.

Register this subset under a new manifest, source and scope. Freeze the exact
three cells, roles, budgets and phase policy before seeking exact authorization.
The user's request permits preparation; no old authorization is reusable.
The two-hour monitor remains paused until this exact attempt is authorized and
started. It then reads only the new root, remains quiet during normal progress,
and reports transitions, completion or actionable failures. Stop requests take
priority. This document does not itself authorize any API call.
