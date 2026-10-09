# MATH Baseline Calibration Stage 1

Execution: `EXECUTION_COMPLETE`. Integrity audit: PASS. Scientific validity: `VALID`.

Frozen source: `6328e5a7e49a672f3da50d85b0245f29dc437f5f`. A/B/C use the same 60 development Optimize examples, seed 81, one Solver member and qwen3-8b with thinking=false. The fixed rotated panel completed 180/180 logical evaluations.

This is an independent baseline calibration. The V2.5 optimizer, old Canary scores, old evidence and production prompts/parsers were not changed. Stage 2, Validation, Test and Shadow were not executed.

## Observed primary results

| Arm | First valid | First correct | Recovered valid | Recovered correct | Successful draws | Charged tokens |
|---|---:|---:|---:|---:|---:|---:|
| A | 23/60 | 23/60 | 23/60 | 23/60 | 171 | 197,431 |
| B | 56/60 | 48/60 | 57/60 | 49/60 | 70 | 71,376 |
| C | 1/60 | 1/60 | 1/60 | 1/60 | 237 | 197,768 |

All primary rates use the 60-example denominator within each Arm. Format-invalid outputs alone trigger up to four semantic draws. A valid wrong answer stops. Truncation alone permits 6144 tokens on the immediately following draw; otherwise the capacity is 3600.

## Paired interpretation

A/B share the native explicit-final Parser and compare Answer instructions. Comparisons involving C change both the instruction and the strict boxed output Parser. They therefore describe the full output interface. Paired improved/regressed counts and descriptive 95% bootstrap intervals (10,000 paired-example resamples, seed 81) are in summary.json.

B versus A improved first native correctness on 27 paired examples and regressed on 2: +41.7 percentage points, descriptive 95% interval +26.7 to +55.0. Recovered native correctness improved on 28 and regressed on 2: +43.3 points, interval +28.3 to +56.7. B first/recovered validity was 56/60 and 57/60. These are development-panel observations.

## Secondary gold-blind diagnostic

| Arm | First verified diagnostic correct | Final verified diagnostic correct |
|---|---:|---:|
| A | 49/60 | 49/60 |
| B | 47/60 | 48/60 |
| C | 46/60 | 48/60 |

The common extractor completed extraction before Gold comparison and ran only after generation terminated. It did not control retries, selection or primary scores. Unknown/unparseable outputs remain separately counted. Its multiple-declaration conflict check can reject an output accepted by the native final-line Parser, so the common diagnostic is not a strict superset of primary validity. Format recovery draws are new stochastic outputs, so diagnostic differences across these draws do not isolate mathematical reasoning improvement.

## Format boundary and recovery cost

A first-draw failures included 30 final labels followed by later content, 3 Markdown labels, 2 truncated outputs, 1 malformed payload and 1 boxed-only output. A repeated 111 extra draws without gaining a native-valid or native-correct example. B used 10 extra draws, gaining one native-valid and native-correct example.

C first-draw shapes were 54 terminal multiline display boxes, 4 boxes with another boundary, 1 truncated output and 1 native-valid box. In the final outputs, 53 were terminal multiline display boxes. The strict preregistered C Parser requires one complete box on the last nonempty physical line; a later display-closing line violates that boundary. Its 177 extra draws gained no native-valid or native-correct example. This boundary incompatibility is verified; C mathematical inability is not established. The C primary score remains 1/60. Detailed post-run shape counts and per-round token costs are separate sanitized files.

## Limits and next step

These 60 stratified development examples exclude the previous 12 Canary examples. The new panel is not directly comparable to the old 12-example five-member Vote. It does not establish population performance, a backend weight revision, or V2.5 search efficacy. Validity and accuracy are separate.

Review B as the independent baseline candidate: it produced 48/60 first-draw and 49/60 recovered native-correct answers with 71,376 tokens, compared with A 197,431 and C 197,768. The first common diagnostic was A 49/60, B 47/60 and C 46/60; the large native-score gap is consistent with output contract compatibility rather than demonstrated mathematical ability improvement. Review the C boundary contract before a larger paid panel. Any revised Stage 2 or C interface requires an explicit amendment, fresh identity and separate authorization. This evidence grants no automatic method promotion, paid expansion or rerun.

## Cost and verification

478 physical requests produced 478 successful responses. The closed ledger charged 466,575 tokens, including 0 conservative unknown-usage tokens, against the independent 1,450,000 charged-plus-reserved ceiling. Terminal reserved tokens: 0. These operational charges do not assert Provider monetary billing.

Offline verification: 444 current tests passed, 2 skipped; 18 execution-specific tests passed. 220 historical/private-artifact modules were omitted by the classified suite; full historical replay PASS is not claimed. compileall, governance, manifest preflight, guarded fake HTTP/ledger tests, original-artifact checks, sanitization and diff checks passed. Offline network attempts were zero.

Public files contain only hashes, counters, categories and metrics. Questions, Gold, prompts, model text, request/response bodies, endpoints, private host paths and raw ledger/checkpoint evidence remain ignored. The API authorization includes no Git push.
