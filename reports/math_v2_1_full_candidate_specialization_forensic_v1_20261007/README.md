# Optimize Full-candidate specialization / collateral forensic

Completed zero-API forensic of the seven attempt6 Full candidates. No scientific method change, A4 rerun, fresh Solver output, provider judge or held-out raw sample access. Parent source is `d294025acc9151e146efae258ac3b9c90d05baa9`; baseline publication is `21388326790ee7a6472c247d84a96d5f84e586ca`.

Initial member correctness sets were independently reconstructed: all five contain the same 22 of 60 examples. Initial Oracle and plurality Vote are both 22/60.

| Candidate | Correct /60 | Retained | Lost | Novel | Oracle /60 | Vote /60 | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| F01 | 19 | 16 | 6 | 3 | 25 | 22 | 3 |
| F02 | 17 | 14 | 8 | 3 | 25 | 22 | 5 |
| F03 | 18 | 16 | 6 | 2 | 24 | 22 | 3 |
| F04 | 17 | 16 | 6 | 1 | 23 | 22 | 3 |
| F05 | 18 | 15 | 7 | 3 | 25 | 22 | 4 |
| F06 | 17 | 15 | 7 | 2 | 24 | 22 | 4 |
| F07 | 19 | 15 | 7 | 4 | 26 | 22 | 3 |

**Q1 — novel coverage:** yes under the frozen evaluator: 18 candidate/example gain events across six previously uncovered examples. Prompt-instance content in F07 and a symbolically equivalent F03 result limit interpretation as reusable reasoning improvement.

**Q2 — competence lost:** every candidate loses 6–8 original correct examples while acquiring 1–4; 47 loss events versus 18 gains. The unsmoothed gain/loss ratios range from 1/6 to 4/7. All lost examples remain correct under each single replacement because four original peers remain.

**Q3 — Oracle versus Vote:** every gain event has one correct vote and four equivalent wrong votes. Exhaustive replacement analysis through the actual equal-weight plurality/tie-abstain aggregator requires two additional correct peer replacements in every event. The result follows from observed answer classes, not an assumed majority threshold.

**Q4 — existing combinations:** actual candidate counts by member are 0,3,0,2,2; including baseline choices gives 36 combinations. Highest Vote remains 22/60. Highest Oracle is 28/60 in T31, whose Vote is 21/60. No beneficial valley crossing is observed among these outputs.

**Q5 — gate counterfactual: MIXED.** Each candidate fails both initial competence floor and strict Vote improvement. Removing either guard alone admits none at Full; removing both admits seven. Oracle-positive intermediate states are excluded, but neither single replacements nor the observed combinations demonstrate Vote improvement. Shadow eligibility and actual commit under altered gates are not evaluable without unobserved Shadow outputs; no such outputs were inspected.

Semantic conclusion: `NO_CLEAR_SPECIALIZATION_SEMANTICS`. Novel coverage and collateral loss are observed; stable coherent specialist roles are not established. See [semantic review](candidate_semantic_forensic.md), [candidate table](full_candidate_specialization_table.csv), [vote topology](novel_coverage_vote_topology.csv), [all combinations](counterfactual_full_candidate_teams.csv), [gate counterfactual](gate_counterfactual.json) and [classification](specialization_collateral_forensic.json).

Pareto comparison is post-hoc and descriptive. Exact prompts, questions, answers, gradients, Pattern and Memory contents remain in an ignored private bundle; its inventory hash is recorded publicly. Public evidence contains only categories, hashes, counters and metrics. Phase A formal network guard reports zero attempted network model calls.
