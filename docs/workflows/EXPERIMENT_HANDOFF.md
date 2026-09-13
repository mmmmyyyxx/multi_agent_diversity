# Frozen Experiment Handoff

This is the required reusable contract between the Sol owner and a scoped Luna
execution subagent. Copy it into an experiment-specific directory and replace
every placeholder before execution. The template itself is not runnable.

```yaml
schema_version: sol_luna_experiment_handoff_v1
experiment_id: REQUIRED

source:
  repository_commit: REQUIRED_FULL_SHA
  tracked_worktree_status: clean

protocol:
  name: REQUIRED
  version: REQUIRED
  protocol_sha256: REQUIRED
  manifest_path: REQUIRED_REPOSITORY_RELATIVE_PATH
  manifest_sha256: REQUIRED
  preregistration_path: REQUIRED_REPOSITORY_RELATIVE_PATH
  preregistration_sha256: REQUIRED

data:
  dataset: REQUIRED
  split_identity: REQUIRED
  split_sha256: REQUIRED

models:
  solver: REQUIRED
  optimizer_or_reflection: REQUIRED
  thinking: REQUIRED
  local_optimizer_backend: REQUIRED
  local_optimizer_version_or_sha256: REQUIRED

execution:
  seeds: [REQUIRED]
  opportunity_budget: REQUIRED
  early_stop_rule: REQUIRED
  exact_runner_command: REQUIRED
  expected_output_directory: REQUIRED_PROJECT_LOCAL_PATH
  expected_checkpoint_path: REQUIRED_PROJECT_LOCAL_PATH
  expected_ledger_path: REQUIRED_PROJECT_LOCAL_PATH
  allowed_retries: REQUIRED

authorization:
  api_scope: REQUIRED
  validation_access_policy: REQUIRED
  test_access_policy: REQUIRED

fail_closed_conditions:
  - frozen identity or hash mismatch
  - unexpected tracked source mutation
  - command, seed, model, split, or budget mismatch
  - provider or credential failure outside frozen retry semantics
  - checkpoint corruption or ledger inconsistency
  - budget overrun
  - unexpected Validation or Test access
  - any need to alter code, protocol, prompts, or scientific choices

READY_TO_RUN: false
```

## Sol freeze checklist

Only Sol may change `READY_TO_RUN` to `true`. Before doing so Sol verifies:

- protocol integrity and preregistration/manifest freeze;
- all recorded hashes and the exact repository commit;
- tracked worktree, split governance, models, budget, and stop rule;
- explicit API authorization and exact Validation/Test policy;
- exact runner command and fresh, project-local artifact destinations.

No frozen handoff means no Luna run.

## Luna pre-run checklist

Luna independently verifies the current HEAD, required files, hashes, clean
tracked source, command, seed/model/budget, access policy, and authorization.
Any mismatch fails closed. Luna must not improvise, repair source, amend the
protocol, change retries, or replace the requested model.

When Luna was explicitly requested but model-selectable dispatch is unavailable,
return `LUNA_DISPATCH_UNAVAILABLE`; do not substitute another Sol.

## Luna execution report

Luna returns operational evidence only:

```yaml
experiment_id: REQUIRED
execution_status: EXECUTION_COMPLETE_OR_EXECUTION_ABORTED
frozen_commit: REQUIRED
command_executed: REQUIRED
opportunities_completed: REQUIRED
termination_reason: REQUIRED
provider_calls: REQUIRED
api_tokens: REQUIRED
checkpoint_status: REQUIRED
ledger_status: REQUIRED
artifact_paths_and_hashes: REQUIRED
validation_calls: REQUIRED
test_calls: REQUIRED
exceptions_or_warnings: REQUIRED
protocol_mismatches: REQUIRED
```

Luna does not label a method better, worse, supported, or failed. After the
handoff returns, Sol performs protocol, artifact, ledger, and split-access
audits and owns all scientific interpretation and publication.
