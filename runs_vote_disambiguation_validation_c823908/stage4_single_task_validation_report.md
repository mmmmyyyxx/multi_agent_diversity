# Stage 4 Single-Task Validation

Scope: `disambiguation_qa`, seeds 42/43/44. The test split was previously observed during method development, so this is a robustness validation rather than a fully untouched final benchmark.

| Setting | Seed 42 | Seed 43 | Seed 44 | Mean +/- SD |
|---|---:|---:|---:|---:|
| Shared baseline | 0.432 | 0.480 | 0.480 | 0.464 +/- 0.028 |
| Full scalar vote | 0.712 | 0.568 | 0.672 | **0.651 +/- 0.074** |
| Guarded diversity | 0.768 | 0.568 | 0.584 | 0.640 +/- 0.111 |

The selected main method beats baseline on all three seeds: +0.280, +0.088, and +0.192. Mean individual accuracy also rises from 0.463 to 0.627, so the improvement does not trade away individual correctness. Paired bootstrap 95% intervals are [0.168, 0.392], [0.024, 0.152], and [0.096, 0.296]. Exact McNemar p-values are 5.13e-6, 0.0192, and 0.000388.

Main-method wrong-to-correct transitions are 47/15/34 versus correct-to-wrong 12/4/10. Every wrong-to-correct case combines increased gold votes with a reduced largest wrong cluster. Mean pairwise double-fault rate is much lower than baseline, and dominant wrong clusters decline on every seed.

The main method wins/ties/loses against baseline on seeds: 3/0/0. Variance is nontrivial, but the effect is not driven by one seed. Expansion to other strict BBH tasks is warranted as a next validation step, not yet as a cross-task generalization claim.
