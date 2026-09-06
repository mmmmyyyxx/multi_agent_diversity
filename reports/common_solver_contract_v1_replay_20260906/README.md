# COMMON_SOLVER_CONTRACT_V1 frozen-artifact replay

The single shared evaluator completed P0 plus the frozen Seed76 MARS, GEPA, and
Diversity final artifacts on ExternalValidation50. No optimization was rerun
and Test50 was not accessed.

| State | Vote | Oracle | Mean member |
|---|---:|---:|---:|
| P0 common | 0.60 | 0.60 | 0.600 |
| MARS final | 0.70 | 0.70 | 0.700 |
| GEPA final | 0.74 | 0.74 | 0.740 |
| Diversity P1 final | 0.66 | 0.92 | 0.588 |

All three frozen finals improve Vote over the shared P0 in this replay. The
evaluation contract is aligned, but end-to-end optimization parity is not: the
artifacts were generated under different historical Solver adapters. A full
parity rerun is justified only if a formal cross-method ranking is required;
the existing within-method gain and saturation evidence remains usable.
