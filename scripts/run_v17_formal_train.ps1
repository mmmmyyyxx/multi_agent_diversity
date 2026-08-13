$ErrorActionPreference = "Stop"
if ($env:V17_FORMAL_TRAIN_AUTHORIZED -ne "1") { throw "V17 formal training API execution is not authorized." }
if ($env:V17_FORMAL_VALIDATION_AUTHORIZED -eq "1" -or $env:V17_FORMAL_TEST_AUTHORIZED -eq "1") { throw "Training launcher does not accept later-phase authorization." }
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = "D:\Anaconda\envs\DL\python.exe"
$RunRoot = Join-Path $Repo "runs\v17_formal_5arm_3seed_20260813"
$FreezePath = Join-Path $RunRoot "phase_a\source_freeze_manifest.json"
$Manifest = Join-Path $Repo "configs\task_level_comparison_strict_bbh_seed42.yaml"
Set-Location -LiteralPath $Repo
if (-not (Test-Path -LiteralPath $FreezePath)) { throw "V17 source freeze is missing." }
$Freeze = Get-Content -LiteralPath $FreezePath -Raw | ConvertFrom-Json
if ($Freeze.source_freeze_status -ne "PASS") { throw "V17 source freeze is not PASS." }
if ((git rev-parse HEAD).Trim() -ne $Freeze.git_head) { throw "V17 execution commit mismatch." }
if ((git status --porcelain=v1 --untracked-files=all | Out-String).Trim()) { throw "Worktree must be fully clean." }
if (Test-Path -LiteralPath (Join-Path $RunRoot "train_protocol_gate.json")) { throw "Training phase already completed." }
$env:V17_FORMAL_SOURCE_FREEZE = $FreezePath
$env:PYTHONPATH = $PSScriptRoot
$Common = @(
  "--workspace", ".", "--manifest", $Manifest, "--tasks", "disambiguation_qa",
  "--dataset_format", "mars", "--out_root", $RunRoot,
  "--train_size", "75", "--val_size", "50", "--test_size", "125",
  "--agent_model", "qwen3-14b", "--optimizer_model", "qwen3-14b", "--evaluator_model", "qwen3-14b",
  "--temperature", "0", "--solver_max_tokens", "1800", "--agents", "5", "--epochs", "1", "--update_every", "10",
  "--proposal_memory_mode", "off", "--num_candidates_per_parent", "2", "--candidate_eval_pool_size", "75",
  "--eval_solver_call_concurrency", "8", "--stage_b_candidate_budget", "2", "--resume_from_checkpoint", "0",
  "--provider_call_budget", "8000", "--total_token_budget", "3000000", "--final_test_enabled", "0",
  "--preserve_final_checkpoint", "1", "--immutable_comparison_cache", "1", "--resume_completed", "0", "--optimized_only", "0"
)
$Orders = @{
  "56" = "shared_static_reference,experimental_v17_formal_generic_2x2_matched,experimental_v16_efficacy_g_matched,experimental_v16_efficacy_r_m20,experimental_v16_efficacy_r_m2f"
  "57" = "experimental_v16_efficacy_g_matched,experimental_v16_efficacy_r_m20,experimental_v16_efficacy_r_m2f,shared_static_reference,experimental_v17_formal_generic_2x2_matched"
  "58" = "experimental_v16_efficacy_r_m2f,shared_static_reference,experimental_v17_formal_generic_2x2_matched,experimental_v16_efficacy_g_matched,experimental_v16_efficacy_r_m20"
}
foreach ($Seed in @("56", "57", "58")) {
  $SeedRoot = Join-Path $RunRoot ("seed" + $Seed)
  & $Python scripts\run_task_level_accuracy.py @Common --out_root $SeedRoot --seeds $Seed --settings $Orders[$Seed]
  if ($LASTEXITCODE -ne 0) { throw "V17 training failed for Seed$Seed." }
}
& $Python scripts\verify_v17_formal_source.py --freeze $FreezePath
if ($LASTEXITCODE -ne 0) { throw "V17 source changed during training." }
& $Python scripts\audit_v17_formal.py --phase train --run_root $RunRoot --freeze $FreezePath --out (Join-Path $RunRoot "train_protocol_gate.json")
if ($LASTEXITCODE -ne 0) { throw "V17 train protocol gate failed." }
