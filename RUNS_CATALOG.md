# Experiment Run Catalog

Audit date: 2026-07-19. The reviewed source of truth is
`run_cleanup_plan.json`; compact records for removed roots are stored in
`run_records/index.jsonl`.

## Retained Roots

| Root | Role | Tasks / seeds | Split evidence | Status |
| --- | --- | --- | --- | --- |
| `runs_task_level_bbh_selected_phase_adaptive` | V1 reward redesign plus baseline | 4 BBH tasks, seed 42 | Historical reused-file protocol | Complete, 8/8 runs |
| `runs_task_level_bbh_tcs_useful_full` | V2 TCS scalar plus baseline | 4 BBH tasks, seed 42 | Historical reused-file protocol | Complete, 8/8 runs |
| `runs_bbh_oracle_pareto_formal_v2` | V3 Oracle Pareto, scalar TCS, and baseline | 4 BBH tasks, seed 42 | Historical task-manifest protocol; root is not uniformly strict | Complete, 12/12 runs |
| `runs_v8_stable_qd_acceptance_fef10f3` | Latest pre-refactor Stable-QD end-to-end and resume smoke | `disambiguation_qa`, seed 42 | Strict split | Complete smoke, 1/1 run |

There is no complete canonical formal root for V7 or Stable-QD V8. The V7
2-task/2-seed pilot and the partial three-seed formal attempt were removed only
after their metrics, costs, metadata, and original paths were preserved in
`run_records/`. They must not be presented as formal evidence.

## Cleanup Result

| Item | Value |
| --- | ---: |
| Run roots before cleanup | 34 |
| Size before cleanup | 1,918,658,485 bytes (1.79 GiB) |
| Retained roots | 4 |
| Removed roots | 30 |
| Released space | 1,377,278,552 bytes (1.28 GiB) |
| Compact deleted-run records | 30 |
| Ambiguous roots deleted | 0 |

The cleanup tool accepts only direct repository children named `runs_*`,
rejects symlinks, traversal, locks, active-process references, and workspace
mismatches, and writes each compact record before deleting its source root.

## Retention Workflow

```powershell
$PY = "D:\Anaconda\envs\DL\python.exe"
& $PY scripts/audit_runs.py --workspace . --output run_cleanup_plan.json
& $PY scripts/prune_runs.py --workspace . --plan run_cleanup_plan.json --dry-run
```

`--apply` is permitted only after manual review of every plan row. An
`AMBIGUOUS` row is always retained. The repository ignores all `runs_*/`
content; only compact `run_records/` summaries are versioned.
