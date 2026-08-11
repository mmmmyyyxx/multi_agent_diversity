# v16 experimental Module2 Seed51 mechanism Pilot

This directory contains the sanitized, analysis-ready evidence from the
single-seed C0/C2/C3 train-only mechanism Pilot executed at source commit
`1f3947c6e69b26666614d8be82607175c76048ee`.

Execution completed for all three preregistered arms: 8/8 updates each,
qwen3-14b with thinking disabled, proposal memory off, no validation, and no
final test. No infrastructure failure was observed. The canonical audit did
not complete because the frozen auditor expected `parent_team_hash` directly
in the Module2 context diagnostic schema. The runtime stores the parent hash in
the update decision and TCS context audit instead. The auditor was not patched
after API execution and the Pilot was not rerun; official certification is
therefore `HOLD`, not `PASS`.

The separate read-only fact check found zero W1, common-safe, max-one-commit,
vote-correct propagation, context serialization, forbidden-field, and
cross-branch repair-duplication violations. Two exact same-parent/target C2/C3
contexts were comparable and had identical Repair/Preservation membership.
These facts do not substitute for the canonical gate.

Mechanism evidence is directional and descriptive only. C0→C2 showed no
improvement: feasible retention fell from 36.4% to 10.0%, while F geometry rose
from 59.1% to 90.0%. C2→C3 was mixed: feasible retention rose to 26.1% and F
fell to 69.6%, but ten C3 candidates lost at least one P1 capability. Train vote
was secondary and moved 51→56 for C0, 51→51 for C2, and 51→57 for C3.

No validation or test evidence is present, and no generalization, statistical
significance, or canonical-v16 promotion claim is made.

Raw prompts, questions, answers, model responses, caches, checkpoints,
credentials, endpoints, and absolute local paths are excluded.
