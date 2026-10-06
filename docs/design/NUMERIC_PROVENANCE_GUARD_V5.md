# Numeric abstraction admissibility V5

The frozen semantic boundary prohibits example-specific numeric leakage. It
does not prohibit numeric characters. Generic constants and structural quantities
are allowed only when necessary to state a reusable corrective reasoning rule.
For example, probability normalization, checking a zero denominator and primitive
integer coefficients may require a small mathematical constant even when the
same value occurs in the example. The 400-character Gradient limit remains.

`PATTERN_ABSTRACTION_SPECIFIC_CONTENT_GUARD_V5` and
`PER_EXAMPLE_TEXTUAL_GRADIENT_PROMPT_V3` change provider-output admissibility and
the extraction prompt only. Gradient V1, one fresh generation per wrong example,
gradients-only clustering V4, same-F, WHO, Layer1, Memory, transition, models,
decoding, membership, budgets and stopping are unchanged. There is no regeneration
or corrective provider call. The cluster prompt remains byte-identical.

The deterministic numeric detector compares each individual Gradient with its
problem, reference and prediction. A generalized Gradient is checked against
the complete selected-member wrong universe, preserving the existing abstraction
boundary. NFKC digit forms, grouped integers, decimals, scientific notation,
English cardinal words and simple integer slash/LaTeX ratios have comparable
values. Exponent expansion is bounded to 600 decimal places. Signs do not supply
provenance by themselves. The detector rejects these stronger signals:

- An explicit answer/return/output assertion of a reference or prediction value.
- A matching long value (magnitude at least 100), a decimal with at least three
  significant digits, or a nontrivial ratio. The simple structural ratios zero,
  one, two and one half alone do not trigger the ratio check.
- A matching value explicitly tied to a given, specified, provided, stated,
  observed or supplied quantity, or with a matching concrete unit/actor quantity.
- A source expression fragment of at least five non-space characters, with a
  numeric value outside zero, one and two, copied with optional whitespace.

A shared small number alone is insufficient, including zero, one and two.
Numeric answer substring matching is delegated to this detector; symbolic-answer
checks, entity contexts, coordinated participant constraints, supplied example
text, quoted labels, explicit expression copying and external-interface checks
remain. The original V3/V4 helpers retain their default historical behavior.

This is a bounded heuristic. It neither proves provenance nor recognizes all
units, languages, formula transformations, number paraphrases or semantic
repetitions. Generic rules with distinctive values can still be false positives;
ambiguous small overlaps can remain false negatives. No comprehensive semantic
safety, gradient usefulness or population compliance rate follows from passing.
Contract compliance and semantic usefulness require separate descriptive audits.

Historical V4/prompt V2 attempts and classifications are immutable. New attempts
use a strict data-only amendment over the hash-pinned initial-condition parent.
Only the named prompt/guard changes and fresh attempt/cache/scope metadata are
admissible. Old authorization, checkpoints and cache realizations cannot be
reused. Current execution rejects old policies; explicit replay uses the old
source and contract. Pilot attempt2 starts with five byte-identical V1_2 minimal
prompts and empty Memory. No new Canary is needed; Validation and Test stay closed.
