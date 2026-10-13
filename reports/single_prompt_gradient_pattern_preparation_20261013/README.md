# Independent Gradient–Pattern single-Prompt preparation

Status: offline implementation; **no real experiment, API authorization or
Validation/Test content access**. This report does not contain empirical method
results. The complete original autonomous task is retained privately by its task
attachment; scientific settings were separately confirmed by the user.

## Audit and decisions

Audit base commit: `d7de59818da609cff6b03c6ebb334fa79011aedd`.
The current method is V2.10 Unified Team Prompt Search. Gradient–Pattern is
active, but its diagnostic universe is **all selected-member Optimize errors**.
The value3 in the current Reflection policy is a representative-support limit,
not a formal error diagnostic Batch. No fixed20-round single-Prompt protocol
exists in that current graph. Historical per-opportunity six-generation settings
are not silently treated as an outer optimization horizon.

The user explicitly approved a separate random wrong-example Batch of3 and then
selected20 rounds. Further confirmations approved raw Batch plus Gradient/Pattern
for C, post-search measurement of all Validation checkpoints, paired seeds88/89/90,
existing MATH/models/initial Prompt, structural-only replacement and exception
handling. The user chose **no cumulative Token hard ceiling** and finally selected
the original repository runtime `runs/math_v2_1/runtime`, with SDK2.37.0.
These settings are design approval only. Real calls require a later exact grant.

The normative [independent protocol](../../docs/design/SINGLE_PROMPT_GRADIENT_PATTERN_V1.md)
and [study manifest](../../experiments/manifests/single_prompt_gradient_pattern_v1.json)
describe the complete scope. The current team graph/defaults and historical
evidence remain unchanged. Registration uses a separate study schema, with nine
independent cell manifests. The shared current runner dispatches this schema to
an isolated research binding; it does not claim to instantiate the current team.

## Reused components and isolation

Reuse the Arm B initial artifact, editable Role/Strategy/Answer value object,
raw-problem Solver interface, MATH scorable V2 split/evaluator, flexible extraction,
four-draw invalid recovery and capacities, role-routed models/generation contracts,
ordered bounded workers, durable hash-chained Token ledger, response receipts,
Gradient prompt/input/field/abstraction validators, semantic clustering prompt,
alias partition recovery and normalized corrective-identity support merging.

New production modules under `multi_dataset_diverse_rl/research/` implement the
independent schedule/update loop, uniform error Batch, isolated inference ports,
study phase/source/runtime/authorization freeze and diversity analysis. No old
runner, team responsibility ranking, focus selection, memory, population, Full,
Probe, Shadow, accuracy gate or best-prompt selection is executed.

A/B use the same Optimizer instruction and call shape, differing in error count.
C uses the same raw Batch as B plus legal diagnostics and all legal Patterns.
C retains the formal labeled4096-character reference preview inside Gradient;
its direct raw feedback still contains complete solutions. Full-Errors delivers
every current error with complete resolved response and reference evidence.
Context excess is a recorded blocker, never sampling or truncation.

Performance is measured after direct replacement. A legal unchanged candidate
uses one draw and receives its own explicit status. No-error and exhausted-format
rounds carry their Prompt and continue. Provider/ledger/authorization/persistence
failures abort a phase with evidence preserved and no automatic retry/resume.

## Planned measurements and resource accounting

| Phase | Membership/checkpoints | Maximum logical Solver evaluations | Maximum physical calls, all roles |
|---|---|---:|---:|
| Train | Nine cells ×21 Optimize150 checkpoints | 28,350 | 114,480 |
| Validation | All21 checkpoints after all nine searches close; Validation300 | 56,700 | 226,800 |
| Test | Nine final Prompts only; Test300 | 2,700 | 10,800 |

These are **ceilings**, not expected cost. Identical paired requests and fewer
errors can reduce calls. Train additionally permits at most180 logical candidate
edits,180 logical per-example diagnoses and60 logical clustering operations.
Physical bounds include four Solver draws, two candidate draws and three
Gradient/Cluster draws. Every request has frozen output limits and a durable
reservation/charge; reliable supplier usage and conservative fallback charges
are reported separately. A meaningful expected total Token cost is not available
before real responses; this report does not substitute a synthetic cost estimate.

Raw Prompt histories, questions, references, responses, diagnostics and per-item
outputs stay in ignored private runs. Sanitized analysis contains all36 pairwise
comparisons and complete9×9 matrices, three within-arm pairs, nine cross-arm
pairs per method contrast (including the three same-seed pairs), correctness
overlap, double-fault, Oracle coverage/gain and individual performance. Co-valid
answer disagreement uses pinned mathematical equivalence, with invalidity
reported separately. Strategy interpretation requires owner review of private
evidence. No efficacy, significance or Pattern-only causal claim is made.

## Verification and reproduction

See [verification.json](verification.json) for the final verification record.
The selected repository runtime uses the pinned MATH dependencies. Tests and deterministic
smokes run through `tests/formal_zero_api_runner.py`, with credentials removed,
pre-import network blocking and a negative control. Earlier default-Python failure
was a missing-dependency diagnosis, not a scoring-contract change. An intermediate
DL current-suite run had742 passes,26 failures and2 skips: all26 failures were
the historical execution binding's exact SDK2.37.0 check against DL's SDK2.3.0.
DL's synthetic HTTP compatibility check passed. The user declined an SDK upgrade
and selected the original runtime instead; historical version guards were retained.

The final selected-runtime current suite has **770 passes and2 skips**; the
focused study suite has **21 passes**. The two skips retain the existing
`USER_DEFERRED_DOWNLOAD` prerequisite. Production compilation, manifest metadata
preflight, repository governance and the deterministic smoke pass with zero
network attempts. The classified suite omits224 historical/private modules and
five Git-only cases; no complete historical replay PASS is claimed.

The nine-cell dry run uses synthetic toy arithmetic only and writes21 Prompts
per cell,189 synthetic Validation checkpoint evaluations and all36 final pairs.
It is execution/control evidence only; none of its scores are real run results.
The execution SHA is recorded in the private frozen phase prep after the final
implementation commit, avoiding a self-referential source SHA in this report.

Reproduction with the selected interpreter:

```powershell
& runs/math_v2_1/runtime/Scripts/python.exe tests/formal_zero_api_runner.py tests/current/test_single_prompt_study.py
& runs/math_v2_1/runtime/Scripts/python.exe tests/formal_zero_api_runner.py --offline-command scripts/run_experiment.py --manifest experiments/manifests/single_prompt_gradient_pattern_v1.json --preflight
& runs/math_v2_1/runtime/Scripts/python.exe tests/formal_zero_api_runner.py --offline-command scripts/dry_run_single_prompt_study.py --destination runs/single_prompt_dry_run_fresh
& runs/math_v2_1/runtime/Scripts/python.exe tests/formal_zero_api_runner.py --offline-command scripts/prepare_single_prompt_study.py --phase train --destination runs/single_prompt_gradient_pattern_v1/train_prep
```

The last command requires clean committed source and creates a **false**
authorization template. Actual train execution remains blocked until the user
authorizes the exact frozen source/scope and nine named attempts. Validation and
Test require their own later freeze, sealed search predecessor and explicit grant.
