# Refactor Metrics

Baseline SHA: `2be2e30eef56d1019fe0dcf3cd6078c7edbf0872`.

| Metric | Before | After |
| --- | ---: | ---: |
| `multi_dataset_diverse_rl/system.py` lines | 10,740 | 41 |
| `TraceBeamSearchSystem` methods defined in `system.py` | 217 | 0 direct algorithm methods; mixin assembly only |
| Largest core module | 10,740 lines | 1,386 lines (`optimization/candidate_generator.py`) |
| `update_prompt_with_beam` | 1,330 lines | 12-line coordinator over 7 explicit stages |
| `cli.py` lines | 1,944 | 1,213 |
| Flat configuration schema fields | 296 | 299 compatibility fields distributed across sections |
| Top-level canonical Config state | flat fields | 11 section objects |
| Experiment setting record fields | 71 optional fields | 3 preset fields: name, base, overrides |
| Checkpoint version | 5 | 6 |
| Run roots | 34 | 4 after cleanup; 5 after the permitted targeted smoke |
| Local run bytes | 1,918,658,485 | 541,379,933 after cleanup; 555,792,229 after smoke |

## Current Modules

The largest modules are below the 1,500-line target. `system.py` assembles
lifecycle, runtime state, candidate schema, solver, metrics, target selection,
candidate generation/evaluation, prompt update, training, joint selection,
dataset evaluation, and artifact mixins. Formulas and state serialization live
in their responsibility modules.

The prompt update pipeline is:

```text
CandidateGenerationStage
CheapPrescreenStage
CandidateEvaluationStage
CandidateClassificationAndRefillStage
ArchiveSelectionStage
CandidateEventStage
UpdateSummaryStage
```

Canonical typed models include `CandidateRecord`, `CandidateMetrics`,
`BehaviorProfile`, `MechanismRepresentation`, `QualityCounts`, `QualityAnchor`,
`JointSelectionResult`, and `LineageState`. Legacy modules re-export canonical
functions where old imports must remain valid.

## Authorized Behavior Changes

1. Specific residual-only mechanisms can enter semantic families; generic
   reasoning or formatting text still fails the specificity gate.
2. Refill checks raw candidates, retained archive niches, and final
   representatives, so archive collisions can trigger additional generation.
3. Quality feasibility uses a frontier of at most five real prompt teams,
   never a component-wise synthetic team.

Characterization tests lock all unrelated behavior. The deterministic
pre-smoke suite passes 344 tests; compileall and `git diff --check` also pass.

## Targeted Smoke

The only post-refactor API smoke used `disambiguation_qa`, seed 42, two epochs,
and 20/20/20 examples. It completed final test and removed its transient
checkpoint. All eight update summaries completed TCS; Student produced 22
usable candidates with no JSON parse failure. One archive collision triggered
two refill rounds. Final per-agent Safe archive sizes were 4/5/5/3/3 and every
joint representative beam had three entries.

Epoch 2 enumerated the theoretical 243 teams, evaluated 163 after the active
change limit, and retained 73 real-anchor-feasible teams. The real anchor
frontier contained one non-dominated actual team and used no fallback. Final
lineages were two uncommitted, two provisional, and one committed. Final test
was vote 0.35, mean individual 0.39, and oracle 0.55. The run made 1,264 LLM
calls and is execution evidence, not an accuracy claim.
