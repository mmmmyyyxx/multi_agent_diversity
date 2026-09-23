# Current and historical code map

The active research engine is in `multi_dataset_diverse_rl/experiment.py`.
`local_optimizers/production_backends.py` exposes GEPA and MARS through one
local optimizer contract. `team_search/` owns Layer-2 evidence, controller,
team admission, and write-back. `saturation.py` owns shared stopping state;
`provider_factory.py` and `governance/` own execution boundaries.

`scripts/run_experiment.py` parses the new manifest shape. Its real execution
gate is currently closed pending a source-frozen governance adapter. Offline
engine fixtures remain available through the Python API.

Historical v15 code, including `system.py`, `tcs.py`, old member-aware
controllers, and `scripts/run_task_level_accuracy.py`, remains in place for
canonical replay. Experiment-named scripts such as
`scripts/run_seed78_primary_responsibility_ab.py`, earlier GEPA canaries, and
the sequential pilot runners remain for their original frozen protocols.
They are not imports of the current engine.

Do not move or delete a historical path merely to shorten the repository tree:
reports and frozen manifests may refer to it. A new experiment uses the typed
engine after a fresh execution freeze, preregistration, and authorization.
