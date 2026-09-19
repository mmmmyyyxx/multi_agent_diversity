# Preparation handoff — NOT EXECUTABLE

Null execution fields explicitly mark the blocked freeze.

```yaml
schema_version: sol_luna_experiment_handoff_v1
experiment_id: local_gepa_acceptance_rate_pilot_v1
source:
  base_commit: ed81635c1526bc57d31d565cb187b04d7b651c5b
  repository_commit: null
  tracked_worktree_status: preparation_changes_not_committed
protocol:
  name: local_gepa_acceptance_rate_pilot_v1
  version: 1
  protocol_sha256: 1942fcd23cd39170fa2e4e45d4d2dcf90f9eaeceb1ef21aa5c5b9dc475d015ea
  manifest_path: experiments/manifests/local_gepa_acceptance_rate_pilot_v1.yaml
  manifest_sha256: f2020914e393639dce5011e929f7a37c840e8981cdc779bf39be59aec7e986a7
  preregistration_path: experiments/local_gepa_acceptance_rate_pilot_v1/PROTOCOL.md
  preregistration_sha256: 7ccd12cbc03f845d44172e85721014c406c2f8b5d8f8454ef2323a7124ce9f82
data:
  split_identity:
    optimize100: anti_overfitting_split_v1_fold_a_plus_b
    shadow50: anti_overfitting_split_v1_fold_c
    validation50: not_accessed
    test50: blocked
  selected_parent_ids: []
models:
  solver: qwen3-8b
  optimizer_roles:
    reflection: qwen3.7-flash
  thinking: false
  temperatures:
    solver: 0.0
    reflection: 0.0
execution:
  exact_runner_command: python scripts/run_local_gepa_acceptance_rate_pilot.py --execute
  expected_output_directory: runs/local_gepa_acceptance_rate_pilot_v1_attempt1
  seeds: []
  opportunity_budget: null
  early_stop_rule: N actual events; S skips; hard metric ceiling
  allowed_retries: existing transport only; no experiment resume
authorization:
  api_scope: none
  validation_access_policy: prohibited
  test_access_policy: prohibited
fail_closed_conditions:
- missing compatible frozen parents
- missing execution commit
- hash mismatch
- missing explicit authorization
- provider or ledger or split failure
READY_TO_RUN: false
```
