# v15 three-seed Development Formal

Development evidence only; the historical test split was previously exposed. This is not an untouched paper-heldout evaluation.

| Seed | Setting | Train vote | Test vote | Test accuracy | Mean member | Min member | Oracle | Train accepted | Train tokens | Test tokens |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 48 | Static | 50/75 | 85/125 | 0.6800 | 0.6800 | 0.6800 | 0.6800 | 0 | 0 | 31557 |
| 48 | S0 | 57/75 | 90/125 | 0.7200 | 0.6992 | 0.6800 | 0.8000 | 4 | 690455 | 91528 |
| 48 | S1 | 59/75 | 88/125 | 0.7040 | 0.6976 | 0.6800 | 0.8240 | 5 | 1161053 | 111585 |
| 48 | S2 | 58/75 | 88/125 | 0.7040 | 0.6960 | 0.6800 | 0.8320 | 7 | 1969066 | 100429 |
| 49 | Static | 50/75 | 85/125 | 0.6800 | 0.6800 | 0.6800 | 0.6800 | 0 | 0 | 31631 |
| 49 | S0 | 56/75 | 86/125 | 0.6880 | 0.6880 | 0.6800 | 0.7600 | 5 | 702795 | 91972 |
| 49 | S1 | 56/75 | 88/125 | 0.7040 | 0.7008 | 0.6800 | 0.8720 | 8 | 1048181 | 176961 |
| 49 | S2 | 58/75 | 89/125 | 0.7120 | 0.7104 | 0.6800 | 0.8560 | 9 | 2050982 | 137960 |
| 50 | Static | 50/75 | 86/125 | 0.6880 | 0.6880 | 0.6880 | 0.6880 | 0 | 0 | 31699 |
| 50 | S0 | 55/75 | 89/125 | 0.7120 | 0.6960 | 0.6880 | 0.7680 | 3 | 472543 | 94028 |
| 50 | S1 | 56/75 | 87/125 | 0.6960 | 0.6992 | 0.6880 | 0.8400 | 7 | 1171765 | 139434 |
| 50 | S2 | 58/75 | 86/125 | 0.6880 | 0.6912 | 0.6640 | 0.8400 | 9 | 1681095 | 140329 |

## Adjacent test vote deltas

- Static->S0: [5, 1, 3]; mean 3.000; positive/zero/negative [3, 0, 0]
- S0->S1: [-2, 2, -2]; mean -0.667; positive/zero/negative [1, 0, 2]
- S1->S2: [0, 1, -1]; mean 0.000; positive/zero/negative [1, 1, 1]

Training artifact mutation count: **0**.
Total train/test/combined tokens: **10947935 / 1179113 / 12127048**.
