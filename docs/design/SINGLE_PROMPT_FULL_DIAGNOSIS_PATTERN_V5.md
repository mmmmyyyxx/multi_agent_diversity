# Single-Prompt full-diagnosis Gradient Pattern ablation V5

This user-requested amendment supersedes the three-error diagnostic Batch for new
single-Prompt default plans. It is preparation, not API authorization or a restart.
Stopped V3/V4 attempts and their evidence remain unchanged and are not full-diagnosis
results. No old checkpoint, cache, Teacher response or authorization is imported by V5.

## Diagnostic universe and evidence path

A receives all current Optimize errors directly. B receives the seeded random
Batch of up to three errors. C receives ALL current Optimize errors (including
invalid outputs), before any sampling. Every C error is passed to the actual formal
GradientExtractor/PerExampleGradientProvider, with the original formal prompts,
parsers, abstraction guards, trajectory provenance and three total draws per item.
Exhausted or nonactionable items are omitted from clustering; surviving Gradients
enter the actual GradientClusterProvider and normalize_gradient_partition together.
Discovery is not capped at three items. Its resource bound is Optimize150.

C selects one Pattern by maximum unique support count, retaining the explicitly
approved Seed+round tie rule. This is the single-Prompt selection adaptation;
the formal team weighted responsibility selector is not used or fabricated.
After selection, the actual production sample_pattern_support implementation
selects at most three support examples by seeded task-type round robin. The team
controller calls this same function with its original salt and unchanged behavior.
The isolated adapter uses seed42, member index0, round minus one and a versioned
single-Prompt root digest of the actual current Prompt hash. No repeat/cell salt
is added: identical Prompt/support/seed/round inputs select identical representatives
across repeats; independent repeat IDs continue to isolate Solver realizations.
No fake team diagnosis, Vote, peers or responsibility labels are introduced.
Optimize task type and level are carried from the real row metadata for sampling.

Optimizer receives the selected Pattern (including its true support membership),
only the selected representative raw errors and their corresponding Gradients and
diagnostics. It does not receive raw nonrepresentative examples, other Gradients,
other Patterns, partition audits or failed diagnostic text. Full diagnosis and
sampling metadata remain in private audit artifacts. Representative selection
occurs after full clustering, never before it. Fewer than three supports are kept
as-is; no padding or examples outside the selected Pattern are allowed.

No legal Pattern, including total Gradient or Cluster exhaustion, invokes the
existing B-style fallback: independently take its seeded random up-to-three-error
Batch from the whole current wrong set and pass structured_feedback=null.
This fallback uses no failed Gradient as a clustering input and does not abandon
an optimization opportunity. An empty error set produces an empty feedback view.

## Unchanged optimizer and operational contracts

All groups use the same single-block optimizer, strict recorded Validation100
improvement by at least one correct answer, twenty opportunities, and initial
candidate plus at most three format/length/no-change retries. Only exhausted
candidate generation abandons an opportunity. Models, SDK2.37.0, output capacities,
scoring, memberships, train Solver32/Gradient8 and final TestSolver16 stay unchanged.
All three repeats use algorithm seed42 and separate repeat IDs. Only Solver terminal
observations share a realization within the same repeat and phase across arms and
rounds; no cross-repeat cache or Teacher response replay is permitted.

The V4 narrowly scoped discarded-length recovery remains bound explicitly:
ordinary truncated Teacher outputs consume bounded generation draws; positive
thinking evidence or missing thinking=false dispatch remains fatal. Network timeout
and connection errors use the accounted unlimited delayed retry policy. Genuine
identity, access, ledger, credential or unknown failures remain fail-closed.

The default nine cells have a training successful semantic/in-flight ceiling216900:
three C repeats times twenty opportunities times (150*3+3) diagnostic draws, plus
all groups' existing Solver and four-draw candidate ceilings. Failed network calls
are excluded from the semantic ceiling and fully accounted; physical failures and
cumulative Tokens have no hard ceiling. Final Test300 has its separate10800 ceiling.
Training cannot read Test. Final Test requires complete audited nine-cell training,
strict gates, receipts, ledger, final Prompts and an explicit fresh phase grant.
Preparation and changed defaults grant no execution or historical evidence reuse.

## Integrity and verification

Plan/kernel/module identities distinguish full diagnosis from historical Batch
runs. Frozen identities include diagnostic universe, representative policy/salt,
actual production source hashes, per-item and per-opportunity bounds, recovery,
cache, data, models and phase access. Closure recomputes the representative view
from the full private diagnosis and verifies exact coverage of the prior wrong set.
Tests cover over-three diagnostic inputs, success-only clustering, largest support,
formal sampling equivalence, representative-only Optimizer inputs, fallback,
exhaustion, strict gate, phase separation, historical preservation and denied reuse.
