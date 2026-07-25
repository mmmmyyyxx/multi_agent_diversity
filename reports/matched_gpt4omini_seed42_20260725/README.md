# GPT-4o-mini Matched Pilot Records

This directory publishes a reviewable, payload-free subset of the completed
matched efficacy pilot.

## Run Identity

- Source commit: `a51843fdab103b96d24c932ca46ba14c4ced09f0`
- Source working tree: clean
- Method: `member_aware_peer_state_v3`
- Task: `disambiguation_qa`
- Seed: `42`
- Models: `gpt-4o-mini` for Solver, Optimizer, and Evaluator roles
- Split sizes: `75 / 50 / 125`
- Settings, in execution order:
  - `shared_baseline`
  - `shared_independent_accuracy`
  - `shared_member_aware_full`

The three settings share the same dataset and question-set hashes, immutable
Solver contract, retry protocol, initial prompt hash, and root-level resolved
Solver cache identity. The public copy does not include the cache itself.

## Directory Layout

`source_records/` contains the exact aggregate result rows and experiment-run
identities. Each setting directory contains:

- validation and update history;
- candidate funnel;
- target-priority and responsibility lifecycle audit;
- TCS context hashes and boundary metadata;
- per-call status, token, latency, retry, and finish metadata;
- Solver recovery and cost summaries;
- a derived final aggregate summary with per-question rows removed;
- a derived run-identity summary with paths, environment fields, and prompt
  text removed.

`source_records/manifest.json` records byte sizes and SHA-256 hashes. Files
marked `exact_source_copy` are byte-for-byte copies of their generated source
artifacts. Files marked `derived_redacted` are structured projections described
above.

## Publication Boundary

The repository does not contain API credentials, raw API responses, response
excerpts, questions, gold answers, member answers, candidate prompts,
Teacher/Critic/Student text, SQLite cache, checkpoints, or local absolute
paths.

This directory is an experiment-record publication, not a statistical claim.
The pilot uses one task and one seed.
