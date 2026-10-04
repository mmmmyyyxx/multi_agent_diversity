# MATH proposal semantic postmortem

The pipeline completed, but no candidate reached team evaluation: this is ZERO_INTERVENTION / NO_ADMISSIBLE_CANDIDATE. The historical NEUTRAL signal is retained; V2.1 efficacy is not identified.

## Frozen evidence and independent replay

Retrospective, zero-API audit of A1 Seed81 Pilot V4. The pinned public GEPA extractor recovered all 10 physical proposals; exact request/response identities, 28 callbacks and 18 chronological cache replays reconciled. Full failed checks were reproduced using the original parents and frozen Optimize evidence. Original reports, raw evidence and the token ledger were unchanged.

## What blocked candidates

| Owner semantic category | Unique proposals | Meaning |
|---|---:|---|
| Output presentation/content control | 6 | Output sections, response-content restrictions or presentation instructions |
| Answer representation guidance | 3 | Requested mathematical form; the proposer explicitly bans answer-format instructions |
| Benign reasoning/goal verification | 1 | No formatting or output-interface instruction |

These are hash-bound annotations by one retrospective owner reviewer, not independent ground truth. The representation boundary is interpretive and reported separately. Under the explicit reasoning-only proposer rubric and remaining guards, 1/10 would be admissible. This finite audit does not establish a general false-positive rate.

No proposal contained the literal wire marker or copied interface literals. The three copied_solver_interface flags came from generic format-directive regexes. All ten Reflection inputs contained zero final-answer phrase matches, so direct copying of that phrase from supplied evidence is not supported. Generating new output-format guidance remains incompatible with the stated component contract.

All 10 proposals hit forbidden_final_answer_marker; 3 also hit copied_solver_interface; **5 also hit append_only**. Logical replay totals were 28, 8 and 12 respectively. Primary-category summaries obscured the append-only constraint.

## Diagnostic interventions

Replacing only the lexical phrase, while retaining every other guard, admitted **3/10**, including **2** owner-classified output-control proposals. It therefore cannot safely repair the boundary. The benign proposal was rejected solely by the lexical phrase check; the other nine have answer-format/output-control guidance under the strict proposer rubric, with overlapping append-only failures. No counterfactual proposal was scored, exported, committed or reused in a real experiment.

| Physical index | Member | Semantic category | Other checks | Logical occurrences | Phrase-mask boundary accepts |
|---|---:|---|---|---:|---|
| 1 | 3 | OUTPUT_PRESENTATION_CONTROL | append_only | 4 | no |
| 2 | 3 | OUTPUT_PRESENTATION_CONTROL | none | 4 | yes |
| 3 | 1 | OUTPUT_PRESENTATION_CONTROL | copied_solver_interface | 3 | no |
| 4 | 1 | OUTPUT_PRESENTATION_CONTROL | none | 3 | yes |
| 5 | 2 | BENIGN_REASONING | none | 3 | yes |
| 6 | 2 | ANSWER_REPRESENTATION_GUIDANCE | copied_solver_interface | 3 | no |
| 7 | 4 | ANSWER_REPRESENTATION_GUIDANCE | append_only | 2 | no |
| 8 | 4 | ANSWER_REPRESENTATION_GUIDANCE | copied_solver_interface, append_only | 2 | no |
| 9 | 0 | OUTPUT_PRESENTATION_CONTROL | append_only | 2 | no |
| 10 | 0 | OUTPUT_PRESENTATION_CONTROL | append_only | 2 | no |

## Cache and scientific interpretation

14 opportunities allocated [2, 3, 3, 4, 2] to members 0–4. All selected maximum discounted Responsibility scores; there was one unchanged parent state. Each member produced two physical proposals, then exact repeated requests reused the same proposals. Cache replay was correct and saved 18 calls; 14 opportunities are not 14 independent searches. Controller changes are not supported by this audit.

All ten Optimizer responses dispatched thinking=false, stopped normally, had complete fences and no repetition pathology; completion usage was 309–495 tokens. Missing reasoning token counts remain null. Generation truncation is not the current bottleneck.

Initial and Final state bytes and all 100 paired provider realizations match. Final Validation used zero new physical generations. VoteAcc=47%, OracleAcc=68%, paired delta=0 and bootstrap CI=[0,0] follow from identical treatment states and realizations. The 21 oracle-correct/vote-wrong rows are descriptive baseline coverage only; they were not fed into method design. Initial competence floor, strict gain, specialization, Shadow and local-rejected transfer were never exercised by admissible candidates.

## Disposition

The demonstrated bottleneck combines proposer/guard semantic mismatch, generated output-format guidance, and append-only replacements. One benign lexical rejection is confirmed under the owner rubric; a blanket regex relaxation would admit contaminated proposals. Production contracts remain frozen, including the current-spec requirement to keep the mutable guard unchanged. A future version must align the proposer and validator boundary, explicitly settle mathematical representation guidance, and preserve interface isolation, replacement and example-copying checks before any newly authorized search. No efficacy gain from such a repair has been measured.

No Solver, Optimizer, Responsibility, GEPA, split, evaluator or controller tuning; no new real run, other seed/arm or Test execution. Search remains closed.

## Verification and accounting

Current suite: 942 passed, 2 skipped, 1468 historical/private cases deselected and not executed. Targeted audit tests: 11 passed. Compileall, governance, frozen dependency replay, network guard and diff checks passed. Two auditor-only repairs are preserved: persisted Validation stage labels, then explicit tests/ collection scope after an old ignored repository clone caused import mismatches. The successful proposal replay was reused after hash verification.

New API calls/tokens: **0 / 0**. Cumulative charge remains **840,243 / 40,000,000**, leaving **39,159,757**.

Detailed sanitized hashes, spans, annotations and per-proposal counterfactual outcomes are in proposal_analysis.json and provenance.json. Private proposal text is not published.
