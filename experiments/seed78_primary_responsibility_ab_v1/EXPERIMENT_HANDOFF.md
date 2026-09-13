# Seed78 execution-only handoff

The implementation and preregistration are complete. The execution agent must
not edit source, manifests, splits, schedulers, budgets, models or output roots.

Frozen identity:

- experiment: `seed78_primary_responsibility_ab_v1`;
- protocol SHA256:
  `854879a50c08817e1d1a074b8a9320c034ebd363c80940f2fd73a2b7b3cc9d2d`;
- Seed78 reuses Seed75 split identity: A+B Optimize100, C Shadow50;
- Solver: `qwen3-8b`, thinking false;
- optimizer/reflection: `qwen3.7-flash`;
- order: A then B, each at most 32 opportunities;
- Validation50: only after both trajectories freeze;
- Test50: zero calls.

The first execution attempt is preserved at
`runs\seed78_primary_responsibility_ab_v1` as a zero-provider-call engineering
HOLD. It must never be resumed, overwritten or analyzed as scientific evidence.

First run the zero-API preparation command on the repaired execution commit. It
creates a private run-local handoff whose `source_freeze.json` pins the exact
40-character commit and every source/split hash:

```powershell
conda run -n DL python scripts\run_seed78_primary_responsibility_ab.py --prepare
```

Then verify the generated
`runs\seed78_primary_responsibility_ab_v1_prep_retry1_authorized1\`
`phase_a_gate.json` says `PASS`, the execution commit equals current `HEAD`, and
the tracked worktree remains clean. The user explicitly reauthorized only this
fresh retry1 on 2026-09-13. No later retry inherits that authorization.

```powershell
$env:SEED78_PRIMARY_RESPONSIBILITY_AB_AUTHORIZED='1'
conda run -n DL python scripts\run_seed78_primary_responsibility_ab.py --execute --prep runs\seed78_primary_responsibility_ab_v1_prep_retry1_authorized1 --run runs\seed78_primary_responsibility_ab_v1_retry1
```

The root must be fresh. Resume, overwrite and automatic retry are forbidden.
On any protocol, source, persistence, infrastructure, model, split,
Validation-timing or Test50 isolation failure, preserve evidence and `HOLD`.
After success, run `--audit` before `--analyze`. Publish only the sanitized
report; never publish raw prompts, questions, answers, responses, credentials,
endpoints, caches or checkpoints.
