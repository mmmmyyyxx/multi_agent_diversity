# Flexible mathematical answer extraction V3

This user-requested presentation amendment is normative for
`unified_team_prompt_search_v2_5_flexible_answer_v1`. It supersedes only the
terminal-line extraction rule in RESPONSIBILITY_FALLBACK_REPAIR_V25.md.
The optimization architecture, Gradient weighting, five members, equal
equivalence plurality, pinned mathematical domain and split governance remain
those of that contract. No Solver message or immutable output instruction is added.

The current parser is MATH_FLEXIBLE_ANSWER_EXTRACTION_V3, with interface
MATH_FLEXIBLE_ANSWER_SYSTEM_INTERFACE_V9 and binding
MATH_FLEXIBLE_ANSWER_EVIDENCE_BINDING_V3. Its complete POLICY enters the binding;
protocol, validity, recovery, response profiles, evidence, composition and cache
identities change together. Historical V2 extraction remains frozen for baseline
replay. Old caches, measured competence, profiles and authorizations cannot enter
the new current graph.

Recognize Final answer / FINAL_ANSWER / Answer labels using colon or equals,
including Markdown headings, bullets, bold labels and inline code. A labeled
payload may appear on the next line. Balanced display math, boxed expressions,
matrix environments and a labeled complete latex/math/text code fence may span
physical lines. Accept terminal unlabeled complete boxed expressions and clear
natural conclusions using Therefore, Thus, Hence, So or The [final] answer is.
An explicit label may be followed by explanatory prose or an equivalent conclusion.
An empty answer heading immediately followed by another answer declaration is a
heading, not a competing empty payload. A named coordinate assignment with three
or more distinct coordinates may present its ordered values as a tuple; require
matching arity and pinned Tuple admission. Duplicate coordinates and mismatched
arity are not discarded. The pinned two-component tuple ambiguity stays unchanged.

Select the last explicit label, otherwise the last terminal conclusion or box.
All explicit labels, and recognized concluding declarations after the selected
label, must be
mathematically equivalent without access to Gold. Earlier intermediate boxes and
natural conclusions do not compete with an explicit final label. A standalone
box followed by continuing working is not a terminal declaration. At most 16
competing declarations are inspected after excluding earlier working; larger sets
fail as ambiguous. Payloads are bounded at
the pinned domain's 2,048 characters and wrapped declarations at 4,096. Boundaries retain the
selected declaration's starting offset so a trailing confirmation does not become
fictitious written reasoning. Full ordinary response evidence remains preserved.

For a natural scalar conclusion, a single-letter constant assignment can project
to its right side. An explicit scalar answer can also be confirmed by that right
side. An explicit equation stays an equation, and equivalent whole equations are
compared before scalar projection. Variable-dependent equations are never reduced
to constants. Normalize outer math/Markdown wrappers, a sentence period, Unicode
minus/pi and unambiguous numeric or explicitly grouped radicals. Matrix environment
names are mathematical syntax. The expression guard rejects prose alternatives
before pinned no-fallback parsing; it never extracts a convenient prefix from
phrases such as an answer with two alternatives.

No arbitrary last-number extraction, Gold-conditioned selection or approximate
correctness tolerance is introduced. Empty, malformed, ambiguous and conflicting
declarations remain invalid. Truncated/non-stop provider responses remain invalid
even with a complete answer. First valid response ends bounded semantic recovery,
including a mathematically wrong response. Validity only decides framing, recovery
and vote eligibility; unchanged pinned Gold-first verification decides correctness.
Incorrect valid answers continue into mathematical failure evidence.

Old reports, scores, ledgers, retries, parser constants, manifests and raw evidence
remain immutable. Compatibility statistics from old responses, if computed, are
explicit counterfactual diagnostics and cannot replace official results or claim
realized API savings. The previous B-five-member strict-parser review cannot run
against changed source. A future baseline requires a separately frozen amended
protocol/source and exact authorization. This implementation task has zero API calls.

Invariants: INV-MATH-FLEXIBLE-ANSWER-001; INV-V24-FRESH-ATTEMPT-001.
