# COMMON_SOLVER_CONTRACT_V1 Seed77 frozen-artifact replay

The single shared evaluator completed P0 plus the frozen Seed77 MARS, GEPA, and
Diversity final artifacts on ExternalValidation50. No optimization was rerun
and Test50 was not accessed.

| State | Vote | Oracle | Mean member |
|---|---:|---:|---:|
| P0 common | 0.62 | 0.62 | 0.620 |
| MARS final | 0.62 | 0.62 | 0.620 |
| GEPA final | 0.70 | 0.70 | 0.700 |
| Diversity P1 final | 0.58 | 0.86 | 0.580 |

The Diversity P0-to-final coalition-depth decomposition is recorded separately:
new/lost coverage = 14/2,
new/lost correct member-votes = 32/42,
and G>=3 cases = 31 -> 29.

Relative to shared P0, 1 frozen finals improve Vote,
1 tie it, and 1 are lower in this replay. The
evaluation contract is aligned, but end-to-end optimization parity is not: the
artifacts were generated under different historical Solver adapters. A full
parity rerun is justified only if a formal cross-method ranking is required;
the existing within-method gain and saturation evidence remains usable.
