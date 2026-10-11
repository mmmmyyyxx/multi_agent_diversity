# Direct readable optimizer evidence amendment

The current method is `unified_team_prompt_search_v2_6_readable_evidence_v2`.
This amendment supersedes the interval-reference input representation in
`PARALLEL_COMPACT_OPTIMIZER_V1.md`. Its concurrency and private timing contract
continues unchanged, under fresh current method, execution, binding, cache,
input, prompt and evidence identities.

Every role must receive sufficient, accurate and understandable evidence for
its own task. Programmatic reconstructability alone does not establish model
usability or preserved working ability. Engineering savings must not make an
LLM execute character slicing, resolve opaque hashes or require an absent tool.

`DIRECT_READABLE_OPTIMIZER_EVIDENCE_V2` sends the provenance-validated
`observed_response` and `visible_solution` as their original text strings to
Gradient and Reflection. No interval object or decoding instruction enters
their requests. Retain duplicated preview text for now. The complete ordinary
response is authoritative; the preview can be bounded, and the instruction
explicitly directs the model to the complete response when the preview is
truncated. A truncated generation is distinct from a truncated preview.
Dataset worked solutions remain separately labeled. Existing reference limits,
Memory projections and split boundaries are unchanged by this restoration.

`STRUCTURED_PARALLEL_READABLE_EXECUTION_V5` retains Solver16, optimizer8 and
Gradient8. Independent examples overlap; recovery within each example remains
serial. Input-order result merge, all-worker-drain clustering, serial Reflection
and atomic team admission remain mandatory. `MONOTONIC_PROVIDER_TIMING_V1`,
output limits, strict parsers, leakage guards, scoring, budget accounting and
stopping remain unchanged. Solver and aggregator receive no optimizer context.

Tests must inspect the actual Gradient and Reflection request boundaries,
including long Unicode responses, truncated previews and unavailable written
steps. Complete response bytes, preview text, parent prompts, reference evidence,
selected Pattern and Memory must reach their intended roles. Synthetic tests
establish these information and execution contracts, not model efficacy.

Future deduplication must present one clearly labeled complete text with directly
readable boundaries and independently verify role usability before adoption.
This amendment does not implement that future representation or shrink evidence.
The former codec is retained only as an offline historical analysis utility;
current runtime must not import or call it.

The new preparation is DRAFT and has no execution or API authorization. Historical
freezes, reports and stopped-run evidence remain immutable. Paid execution requires
a fresh source/protocol/phase freeze and exact single-use user authorization.
