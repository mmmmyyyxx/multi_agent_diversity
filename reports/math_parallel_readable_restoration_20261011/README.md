# Direct readable optimizer evidence restoration

Method: `unified_team_prompt_search_v2_6_readable_evidence_v2`.
Gradient and Reflection again receive original complete ordinary Solver response
and preview strings. The instruction labels the full response and bounded preview,
distinguishes preview truncation from generation truncation, and separates dataset
worked solutions from Solver observations. No interval objects or character slicing
instruction reaches the current optimizer boundary. Duplicate preview text is
retained intentionally to prioritize role usability over input savings.

Solver16, Gradient8, serial per-example recovery, ordered merging, cluster barrier,
private request timings and all scoring, guards and stopping rules are retained.
The former input-savings measurements belong to the previous representation and
do not describe this restoration. Full runtime source, binding, cache, input and
prompt identities are fresh. Prior evidence, freezes and consumed scopes are unchanged.

The audit established programmatic losslessness of the old encoding, not preserved
model understanding. This repair removes the model slicing task; it does not claim
real-model diagnostic quality, repair efficacy or generalization. Existing reference
prefix and Memory limits are unchanged, and Reflection's prior reference-step input
boundary is unchanged. Future deduplication requires directly readable complete text
and independent role usability evidence.

Preparation remains DRAFT with no source freeze, paid API grant, Validation or Test
access. Verification status is recorded in `verification.json`.

Current suite: **725 passed, 2 skipped**, with no
failures. Actual role-boundary/concurrency/timing checks: **18 passed**. These
include long Unicode responses, bounded previews, final-only responses and
invalid extraction observations. Compileall, strict DRAFT manifest, source
closure, repository governance, historical schema preservation and sanitization
passed. All 3,784 sealed files retained their original bytes. The 220 historical
or private modules are explicitly omitted, not claimed as passing replay.

The initial targeted check was started while its assertions were being completed;
its invalid-response case incorrectly required a final declaration in an invalid
response. The corrected assertion checks explicit invalidity and original text.
The historical-schema comparison also needed to account for the preceding compact
method's retained generic-discriminator exclusion. Historical branches are unchanged.
The final role-boundary check and current suite use the completed assertions.
No production parser, validity guard or output acceptance was relaxed.
