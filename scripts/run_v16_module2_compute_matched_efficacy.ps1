$ErrorActionPreference = "Stop"

if ($env:V16_MODULE2_EFFICACY_AUTHORIZED -ne "1") {
    throw "Module2 compute-matched efficacy API execution is not authorized."
}

$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = "D:\Anaconda\envs\DL\python.exe"
$PrepRoot = Join-Path $Repo "runs\v16_module2_efficacy_prep"
$OutputRoot = Join-Path $Repo "runs\v16_module2_efficacy"
$FreezePath = Join-Path $PrepRoot "source_freeze_manifest.json"
$Manifest = Join-Path $Repo "configs\task_level_comparison_strict_bbh_seed42.yaml"
$Settings = "experimental_v16_efficacy_g_matched,experimental_v16_efficacy_r_m20,experimental_v16_efficacy_r_m2f"

Set-Location -LiteralPath $Repo
if (-not (Test-Path -LiteralPath $FreezePath)) { throw "Source freeze manifest missing." }
$Freeze = Get-Content -LiteralPath $FreezePath -Raw | ConvertFrom-Json
if ((git rev-parse HEAD).Trim() -ne $Freeze.git_head) { throw "Frozen execution commit mismatch." }
if ((git status --porcelain=v1 | Out-String).Trim().Length -ne 0) { throw "Tracked worktree is dirty." }
if ($Freeze.source_freeze_status -ne "PASS") { throw "Source freeze is not PASS." }
if (Test-Path -LiteralPath $OutputRoot) { throw "Formal output root must be fresh." }
$env:V16_M2F_ONLINE_SOURCE_FREEZE = $FreezePath
$env:PYTHONPATH = $PSScriptRoot

$Common = @(
    "--workspace", ".", "--manifest", $Manifest,
    "--tasks", "disambiguation_qa", "--settings", $Settings,
    "--seeds", "53,54,55", "--dataset_format", "mars", "--out_root", $OutputRoot,
    "--train_size", "75", "--val_size", "50", "--test_size", "125",
    "--agent_model", "qwen3-14b", "--optimizer_model", "qwen3-14b", "--evaluator_model", "qwen3-14b",
    "--temperature", "0", "--solver_max_tokens", "1800",
    "--allow_legacy_setting", "0", "--allow_auxiliary_setting", "0",
    "--agents", "5", "--epochs", "1", "--update_every", "10",
    "--proposal_memory_mode", "off", "--num_candidates_per_parent", "2",
    "--candidate_eval_pool_size", "75", "--eval_solver_call_concurrency", "8",
    "--stage_b_candidate_budget", "2", "--resume_from_checkpoint", "0",
    "--provider_call_budget", "8000", "--total_token_budget", "3000000",
    "--final_test_enabled", "0"
)

& $Python scripts\preflight_member_aware.py --allow_dirty 0 @Common
if ($LASTEXITCODE -ne 0) { throw "Formal preflight failed before API." }
& $Python scripts\run_task_level_accuracy.py --resume_completed 0 --optimized_only 0 @Common
if ($LASTEXITCODE -ne 0) { throw "Formal Module2 efficacy execution failed." }
