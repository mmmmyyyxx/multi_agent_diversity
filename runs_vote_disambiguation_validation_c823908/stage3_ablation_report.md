# Stage 3 Reward Ablation

Scope: single-task validation on `disambiguation_qa`, seed 42. Validation metrics are primary; test metrics are exploratory because this split was already observed during method development.

| Setting | Val vote | Val individual | Val margin | Test vote | Test individual |
|---|---:|---:|---:|---:|---:|
| Accuracy only | 0.660 | 0.680 | 0.360 | 0.664 | 0.626 |
| Guarded diversity | 0.740 | 0.708 | 0.424 | 0.768 | 0.720 |
| Full vote | **0.780** | **0.712** | 0.424 | 0.712 | 0.682 |
| No margin | 0.740 | 0.692 | 0.392 | 0.672 | 0.667 |
| No boundary | 0.700 | 0.716 | **0.432** | 0.728 | 0.694 |

## Findings

- Full vote is selected as the main method because it has the best validation vote accuracy (0.780) and a strong validation individual accuracy (0.712).
- Vote-oriented reward outperforms accuracy-only by 0.120 validation vote accuracy and 0.032 validation individual accuracy.
- Removing margin reduces validation vote accuracy by 0.040 and validation margin by 0.032. Margin changes are nonzero for 82.2% of full-vote candidates, versus nonzero vote changes for 69.4%, so margin is the denser signal.
- Removing boundary reduces validation vote accuracy by 0.080. Full vote has positive boundary gain on 10.0% of candidates, so this term has an observable but sparse contribution.
- Guarded diversity is the strongest matched non-vote control: it exceeds accuracy-only on both validation vote and individual accuracy.
- Every optimized method exceeds the shared baseline test mean individual accuracy of 0.456; the 0.005 degradation constraint is satisfied. This comparison is exploratory because the baseline has no validation phase.

All three new TCS runs passed provenance and delta-consistency audit. One no-boundary TCS group failed to produce candidates after the configured recovery path; it is explicitly logged and does not create an incomplete provenance group.
