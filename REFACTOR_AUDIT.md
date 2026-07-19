# Refactor Audit

Audit baseline: `main` at `a2a479dfce07ad4774f69d3f30008f8842b13bfb`.

This audit is intentionally conservative. A field is not classified as dead
until direct access, `getattr`, dataclass serialization, dynamic CLI mapping,
checkpoint restore, run metadata, analysis scripts, and tests have all been
checked.

## Configuration Inventory

| Area | Definition and consumers | Classification | New home | Compatibility |
| --- | --- | --- | --- | --- |
| Dataset and task | `Config` lines 12-29; CLI, task manifests, runner, metadata | ACTIVE_BEHAVIOR | `DatasetConfig` | Flat aliases retained |
| Models and generation | `Config` lines 18-20, 84-105, 219-225; TCS, API client, cost logs | ACTIVE_RUNTIME | `ModelConfig` and `TCSConfig` | Flat aliases retained |
| Training lifecycle | `Config` lines 31-55, 226-258; CLI, checkpoint, runner | ACTIVE_RUNTIME | `TrainingConfig` | Flat aliases retained |
| Reward schedule | `Config` lines 56-83; candidate evaluator and schedule | ACTIVE_BEHAVIOR | `RewardConfig` | Legacy reward fields retained |
| V7 residual controls | `Config` lines 110-135; selectors, trajectory guards, logs | ACTIVE_BEHAVIOR | `ResidualConfig` | Legacy fields retained |
| V8 competence controls | `Config` lines 136-178; fixed probe, target selector, guards | ACTIVE_BEHAVIOR | `CompetenceConfig` | Legacy fields retained |
| Stable QD and lineage | `Config` lines 179-210; QD archive, joint selector, checkpoint | ACTIVE_BEHAVIOR | `StableQDConfig` | Legacy fields retained |
| Candidate evaluation/cache | `Config` lines 236-249; evaluator, cost, resume | ACTIVE_RUNTIME | `EvaluationConfig` | Flat aliases retained |
| Trace diagnostics | `Config` lines 212-218; metrics only | OUTPUT_ONLY except evaluator toggle | `DiagnosticsConfig` | Flat aliases retained |
| Experiment setting overrides | `ExperimentSetting` lines 6-72 and runner dynamic application | DUPLICATED | named preset overrides | Existing settings remain names/aliases |
| Version strings | config, experiment setting, metadata, checkpoint | ACTIVE_BEHAVIOR | `MethodPreset` | Existing string values preserved |

## Repeated or Coupled Fields

- `beam_size` currently means both retained per-agent archive size and joint
  representative count. The search-loop supplement requires these to become
  separate values while keeping `beam_size` as the representative alias.
- `mechanism_signature_*`, normalized mechanism representation, and the new
  mechanism embedding cache describe one concept in three layers. The public
  legacy signature remains, while new code should use one typed mechanism
  representation.
- Candidate quality is represented as free-form `metrics` dictionaries in
  candidate evaluation, archive selection, update logs, and checkpoint beams.
  A typed candidate assessment is required before deleting any key.
- Probe summaries, validation metrics, final metrics, CSV exports, and
  `history.json` repeat overlapping fields. A shared metrics schema must own
  normalized rates and raw counts.
- The V8 setting is an active preset, not a new setting. It must be migrated
  by translating preset values, not by renaming the public setting.

## Function and Metric Duplication

| Concept | Current locations | Refactor action |
| --- | --- | --- |
| Candidate deltas and coverage transitions | `system.py`, V8 helper tests, rollout summaries | Extract pure count/rate transition helpers |
| Plurality diagnostics | `utils.py`, rollout metrics, joint QD team metrics | Keep `utils.py` canonical and inject into team metrics |
| Team coverage and individual summaries | `_summarize_rollout_rows`, `quality_diversity.team_quality_metrics`, competence probe | Create shared team metric builder with both counts and rates |
| Behavior profiles and distance | `behavior_profiles.py`, joint team code | Keep prompt-static answers separate from team-dependent profiles |
| Mechanism similarity | legacy signature helpers plus `mechanisms.py` | Keep legacy adapter; use normalized representation internally |
| Guard feasibility | candidate guard, Pareto feasibility, QD selection | Split into candidate safe/probation/catastrophic classification and team quality floor |
| Archive selection | legacy Pareto beam and QD archive | Preserve legacy path; isolate Stable QD safe/probation archive policy |
| Serialization | CLI checkpoint, system state, prompt history, result CSV | Introduce typed state records with adapter serializers |

## Large-Module Responsibilities

`system.py` currently combines model I/O, rollout cache, metrics, TCS
generation, candidate evaluation, archive policy, joint selection, lineage,
logging, and persistence. The target split is:

```text
models.py / rollout_cache.py       API and cached solver calls
metrics.py                         canonical team and transition metrics
candidate_policy.py                safe/probation/refill classification
archive_policy.py                  long QD archive and representatives
joint_selection.py                 count floors, bands, folds, active choice
lineage.py                         lineage state machine
system.py                          orchestration only
state.py                           checkpoint/history/run metadata records
```

The first implementation pass may leave compatibility adapters in `system.py`.
No caller-facing behavior may change without characterization tests.

## Serialization and Output Risks

- `checkpoint_behavior_config` is behavior-critical; every new search-loop
  field must enter its fingerprint.
- `run_meta.json`, checkpoint state, prompt history, and task-level CSV must
  derive from the same typed records rather than independently enumerating
  keys.
- JSONL histories are append-only. Versioned record schemas are safer than
  deleting old keys.
- Existing `runs_*` directories are input evidence only and must not be
  modified or committed by this refactor.

## Characterization Plan

Before structural migration, tests cover canonical plurality, mechanism
distance, behavior distance, QD retention, joint team selection, lineage,
checkpoint compatibility, TCS integrity, and validation selection. The search
loop supplement adds dedicated tests for refill, probation, two folds,
hierarchical bands, count floors, change limits, QD readiness, wrong-answer
dispersion, and selector fairness.
