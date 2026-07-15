# Final Report

## Scope And Integrity

- Commit: `c823908f20071c56884f6c2e557b9992d49df365`.
- Tests: 156 passed.
- Task: `disambiguation_qa` only. No other task was resumed, changed, or counted.
- Strict splits: opt 75, validation 50, test 125; all pairwise overlaps are zero.
- Candidate evaluation source: `optimization_train`; validation uses `vote_first`; test is final evaluation only.
- New runs were written only under this root. Historical and failed runs were preserved.

## Four Stages

1. Stage 1 reused three compatible smoke runs with zero calls. Scalar and Pareto both reached vote 0.575; Pareto had higher oracle (0.775 vs 0.750).
2. Stage 2 reused baseline/scalar/Pareto. Selector replay disagreed on 39.3% of updates. Scalar was selected because validation vote/individual accuracy were 0.780/0.712 versus Pareto 0.700/0.672; test was not primary.
3. Stage 3 selected full scalar vote reward. Validation vote was 0.780 versus accuracy-only 0.660, guarded 0.740, no-margin 0.740, and no-boundary 0.700. Guarded was selected as the matched non-vote control.
4. Stage 4 evaluated baseline, main vote, and guarded control over seeds 42/43/44. Main vote averaged 0.651 versus baseline 0.464 and won all three seeds.

## Required Answers

- **A. Scalar vs Pareto:** They make materially different choices (39.3% replay disagreement). Scalar had stronger validation accuracy in this task.
- **B. Vote reward vs accuracy-only:** Yes in this single-task validation: +0.120 validation vote accuracy for full vote.
- **C. Margin density:** Yes. Full vote had nonzero margin delta on 82.2% of candidates versus nonzero vote delta on 69.4%.
- **D. Boundary contribution:** Observable but sparse. Positive gain appeared on 10.0% of full-vote candidates; removing it reduced validation vote from 0.780 to 0.700.
- **E. Individual accuracy:** Not harmed. Three-seed mean rose from 0.463 to 0.627.
- **F. Stability:** The main method beat baseline on 3/3 seeds, though its standard deviation (0.074) shows meaningful seed sensitivity.
- **G. Expansion:** Worth validating on additional strict BBH tasks. These results do not establish BBH-wide or cross-task generalization.

## Statistical Summary

Per-seed main-minus-baseline vote differences are +0.280, +0.088, and +0.192. Paired bootstrap intervals exclude zero for all seeds. McNemar tests are significant at 0.05 for all three seeds. Wrong-to-correct transitions exceed correct-to-wrong transitions on every seed. Improvements jointly increase gold votes and reduce the dominant wrong cluster.

## Artifacts

Core outputs are `stage3_ablation_metrics.csv`, `stage4_results_by_seed.csv`, `stage4_summary.csv`, `paired_bootstrap.csv`, `mcnemar_results.csv`, `vote_transition.csv`, `margin_stratified_analysis.csv`, `accuracy_diversity_decomposition.csv`, `pairwise_error_analysis.csv`, `stage4_process_metrics.csv`, and the stage reports/JSON completion markers in this directory.

Known incompleteness is preserved in `PRIOR_STAGE3_FAILED.json`; it is not included in results. TCS groups that exhausted candidate generation are explicitly logged, while every candidate-producing group passed provenance and delta-consistency audit.
