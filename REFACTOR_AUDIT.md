# Refactor Audit

Audit baseline: `main` at `2be2e30eef56d1019fe0dcf3cd6078c7edbf0872`.

The baseline has 296 flat `Config` fields, 71 `ExperimentSetting` fields,
217 methods on `TraceBeamSearchSystem`, and 10,740 lines in `system.py`.
The generated inventories are `REFACTOR_FIELD_MATRIX.csv` and
`REFACTOR_FUNCTION_MATRIX.md`; regenerate them with
`python scripts/generate_refactor_audit.py` after each structural stage.

This audit is intentionally conservative. A field is not classified as dead
until direct access, `getattr`, dataclass serialization, dynamic CLI mapping,
checkpoint restore, run metadata, analysis scripts, and tests have all been
checked.

## Configuration Inventory

| Area | Definition and consumers | Classification | New home | Compatibility |
| --- | --- | --- | --- | --- |
| Dataset and task | CLI, task manifests, runner, metadata | ACTIVE_RUNTIME | `DataConfig` | Flat properties retained |
| Models and API roles | TCS, solver/evaluator clients, cost logs | ACTIVE_RUNTIME | `ModelConfig` | Flat properties retained |
| Training lifecycle | CLI, checkpoint, runner | ACTIVE_RUNTIME | `RuntimeConfig` | Flat properties retained |
| Candidate generation | TCS, one-shot generator, retry/repair | ACTIVE_POLICY | `CandidateGenerationConfig` | Flat properties retained |
| Candidate evaluation/cache | evaluator, probe cache, cost, resume | ACTIVE_RUNTIME | `CandidateEvaluationConfig` | Flat properties retained |
| Reward and candidate guards | candidate evaluator and schedule | ACTIVE_POLICY | `QualityGuardConfig` | Flat properties retained |
| Stable-QD archive/refill | archive, representatives, parent policy | ACTIVE_POLICY | `ArchiveConfig` | `beam_size` becomes representative alias |
| Joint team selection | enumeration, quality bands, fold stability | ACTIVE_POLICY | `JointSelectionConfig` | Flat properties retained |
| Lineage | lineage state machine and checkpoint | ACTIVE_POLICY | `LineageConfig` | epoch aliases migrate to snapshots |
| Output and diagnostics | histories, model diagnostics, run metadata | OUTPUT_ONLY | `OutputConfig` | Flat properties retained |
| Experiment settings | runners and metadata | DUPLICATED | `ExperimentPreset.overrides` | Existing setting names preserved |
| Method identity | preset registry, metadata, checkpoint | ACTIVE_POLICY | `MethodIdentity` | Existing version strings preserved |

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

Compatibility wrappers may remain in `system.py`, but formulas and storage
details must not. No caller-facing behavior may change without the snapshots
in `tests/test_refactor_characterization.py`, except for the three explicitly
authorized Stable-QD search-space fixes.

## Stable-QD Blocking Findings

1. Residual-only specific mechanisms are normalized into semantic text but
   `mechanism_alternative` prescreen still requires canonical operations.
2. Refill requirements inspect raw evaluated candidates before archive niche
   compression and representative selection, so retained search capacity can
   remain underfilled.
3. `joint_quality_anchor_metrics` is a component-wise maximum assembled from
   different teams and therefore can describe a team that never existed.

These are behavior changes, not refactor cleanup. Their tests and commits must
remain separate from structural migration.

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

## Stable-QD Search-Loop Supplement

The current V8 implementation keeps the public setting
`shared_vote_tcs_competence_depth2_progressive_residual_hybrid` and
`method_version=v8_stable_qd_lineage`. It does not introduce a new setting or
rename the historical V1-V7 refresh path.

The V8 orchestration now has these explicit boundaries:

- Legacy per-epoch `refresh_all_prompt_beams` remains a compatibility path for
  older method versions, but V8 rejects it even when the old CLI flag is set.
- Stable-QD joint selection is event-driven. It skips when there is no material
  archive, representative, or active-prompt change, runs at the configured
  interval, and is forced on the final epoch.
- Fixed-probe work is incremental: active/current representatives and a bounded
  dirty shortlist are evaluated, while completed prompt-question profiles are
  reused. The `3^5` team enumeration is offline and makes zero team-level
  solver calls.
- Candidate generation has separate TCS repair and open mechanism-exploration
  channels. TCS is used when repair evidence exists; otherwise two open
  candidates are requested. Both channels share the same downstream quality,
  archive, refill, and lineage gates.

The new policy fingerprints and counters are persisted in run metadata,
checkpoint state, JSONL diagnostics, and cost summaries. The characterization
tests cover the V8 refresh guard, dual-channel routing, event skip path, and
checkpoint parent-directory recovery. External API smoke runs remain evidence
only and are not committed with the source tree.
