# 42 个 proposal 逐行审计

Local Δ 分别相对机会初始 root 和生成时 incumbent；— 表示未测。Context：E 为显式 optimizer 上下文，O 为可选 reference-value 分支，N 为无显式依赖。分类含单人判断，不能证明 Solver 执行。

| 候选 | Local/4 | Δroot | fixed/lost | Δincumbent | Export | Probe Δtarget | Full/60 | Context | 契约 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| O1G1 | 1 | 0 | 0/0 | 0 | YES | 0 | — | E | PASS |
| O1G2 | 2 | 1 | 1/0 | 1 | YES | 1 | — | N | PASS |
| O1G3 | 1 | 0 | 0/0 | -1 | NO | — | — | E | PASS |
| O1G4 | 2 | 1 | 1/0 | 0 | YES | 1 | 18 | O | PASS |
| O1G5 | — | — | — | — | NO | — | — | N | invalid_structure |
| O1G6 | 2 | 1 | 1/0 | 0 | YES | 1 | 19 | O | PASS |
| O2G1 | 1 | 0 | 0/0 | 0 | YES | 0 | — | O | PASS |
| O2G2 | 1 | 0 | 0/0 | 0 | YES | 0 | — | O | PASS |
| O2G3 | 1 | 0 | 0/0 | 0 | YES | 0 | — | O | PASS |
| O2G4 | 1 | 0 | 0/0 | 0 | YES | 0 | — | O | PASS |
| O2G5 | 0 | -1 | 0/1 | -1 | NO | — | — | O | PASS |
| O2G6 | — | — | — | — | NO | — | — | O | external_output_interface_mutation |
| O3G1 | — | — | — | — | NO | — | — | E | external_output_interface_mutation |
| O3G2 | 1 | 0 | 0/0 | 0 | YES | 0 | — | E | PASS |
| O3G3 | 1 | 0 | 0/0 | 0 | YES | 0 | — | E | PASS |
| O3G4 | 1 | 0 | 0/0 | 0 | YES | 0 | — | E | PASS |
| O3G5 | 1 | 0 | 0/0 | 0 | YES | 0 | — | N | PASS |
| O3G6 | — | — | — | — | NO | — | — | E | external_output_interface_mutation |
| O4G1 | 1 | 0 | 0/0 | 0 | YES | 0 | — | N | PASS |
| O4G2 | 1 | 0 | 0/0 | 0 | YES | 0 | — | E | PASS |
| O4G3 | 1 | 0 | 0/0 | 0 | YES | 0 | — | N | PASS |
| O4G4 | 1 | 0 | 0/0 | 0 | YES | 0 | — | N | PASS |
| O4G5 | 1 | 0 | 0/0 | 0 | NO | — | — | N | PASS |
| O4G6 | 0 | -1 | 0/1 | -1 | NO | — | — | O | PASS |
| O5G1 | 0 | -1 | 0/1 | -1 | YES | -1 | — | E | PASS |
| O5G2 | — | — | — | — | NO | — | — | E | invalid_structure |
| O5G3 | 1 | 0 | 0/0 | 0 | YES | 0 | — | E | PASS |
| O5G4 | — | — | — | — | NO | — | — | E | external_output_interface_mutation |
| O5G5 | 1 | 0 | 0/0 | 0 | YES | 0 | — | E | PASS |
| O5G6 | — | — | — | — | NO | — | — | E | external_output_interface_mutation |
| O6G1 | 1 | 0 | 0/0 | 0 | YES | 0 | — | N | PASS |
| O6G2 | — | — | — | — | NO | — | — | N | fixed_answer_payload |
| O6G3 | 1 | 0 | 0/0 | 0 | YES | 0 | — | N | PASS |
| O6G4 | 1 | 0 | 0/0 | 0 | YES | 0 | — | E | PASS |
| O6G5 | 1 | 0 | 0/0 | 0 | YES | 0 | — | E | PASS |
| O6G6 | 0 | -1 | 0/1 | -1 | NO | — | — | N | PASS |
| O7G1 | 1 | 0 | 0/0 | 0 | YES | 0 | — | E | PASS |
| O7G2 | — | — | — | — | NO | — | — | N | external_output_interface_mutation |
| O7G3 | 1 | 0 | 0/0 | 0 | YES | 0 | — | E | PASS |
| O7G4 | 0 | -1 | 0/1 | -1 | NO | — | — | E | PASS |
| O7G5 | 1 | 0 | 0/0 | 0 | YES | 0 | — | E | PASS |
| O7G6 | 1 | 0 | 0/0 | 0 | YES | 0 | — | E | PASS |
