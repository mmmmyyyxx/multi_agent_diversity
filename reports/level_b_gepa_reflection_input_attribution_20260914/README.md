# Level-B GEPA reflection-input attribution

Gate: **PASS**

Classifier: **SIDE_INFO_REPRESENTATION_CONTAMINATION_IDENTIFIED**

The frozen reflection template and current `decision_procedure` were contract-safe. The historical reflective dataset placed raw Solver responses into `<side_info>`; strict-valid responses necessarily contained the immutable answer marker. It also repeated controller wording about the output interface. This identifies `<side_info>` as the pre-proposal contamination source for the three-call canary.

Exact historical effective-prompt bytes are `NOT_RECOVERABLE_ZERO_API` because request bodies and all nine raw reasoning traces were not durably persisted. The attribution uses frozen source structure, lineage score counts, strict Solver semantics, and the sanitized rejection audit; it does not publish prompts, questions, answers, or responses.

The compatible repair is component-specific evidence: problem text, reasoning-only trace, coarse correctness outcome, and allowlisted reasoning focus. Gold labels, raw failure codes, free-form controller/output-contract wording, and immutable answer lines must not enter reflection side information.
