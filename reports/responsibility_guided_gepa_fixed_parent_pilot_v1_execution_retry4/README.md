# Responsibility-Guided GEPA fixed-parent pilot

Six frozen Optimize100 parent states were evaluated analytically. All branches have actual commit = 0; Validation, ExternalValidation and Test50 are excluded. Raw prompts, questions, responses and endpoint details remain only in ignored runtime evidence.

| Arm | Generated | Valid | Promoted | Full eval | Feasible | Would commit |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 12 | 12 | 12 | 12 | 10 | 6 |
| A1 | 12 | 12 | 5 | 5 | 5 | 5 |
| B0 | 12 | 0 | 0 | 0 | 0 | 0 |
| B1 | 12 | 0 | 0 | 0 | 0 | 0 |

The durable execution gate passed: 6/6 cases, 24 candidates, 12 logical and 12 successful reflection calls, 12 physical reflection attempts, zero commits, zero Validation/Test50 calls, and exact runtime-to-ledger reconciliation.

The scientific result is negative for the current RG-GEPA reflection operator. All 12 reflection outputs failed the immutable output-contract hard gate, so neither B arm reached minibatch or full candidate evaluation. Consequently, reflection quality is not supported and the B0/B1 selector contrast is not evaluable.

Progressive evaluation on the unchanged historical pool reduced full evaluations from 12 to 5 and retained five feasible/would-commit outcomes, versus six would-commit outcomes under full evaluation. This is descriptive evidence of cost reduction with one missed update opportunity, not evidence that RG-GEPA improved proposal quality.
