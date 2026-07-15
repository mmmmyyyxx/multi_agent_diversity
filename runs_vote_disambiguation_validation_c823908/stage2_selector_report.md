# Stage 2 Selector Reuse and Replay

All three `disambiguation_qa` runs were reused from protocol v4; model calls made: 0.

| Selector | Best val vote | Best val individual | Test vote | Test individual |
|---|---:|---:|---:|---:|
| scalar | 0.780 | 0.712 | 0.712 | 0.682 |
| Pareto | 0.700 | 0.672 | 0.648 | 0.624 |

Replay used 56 complete update-attempt candidate pools. Top-1 disagreement was 22/56 (39.3%). Mean replay vote loss was 0.0143 for scalar and 0.0048 for Pareto.

Selected main selector: `scalar_reward`. Scalar has higher best validation vote accuracy; test was not used as the primary selector.

This is single-task validation evidence only.
