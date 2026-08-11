$ErrorActionPreference = "Stop"

if ($env:V16_PILOT_AUTHORIZED -ne "1") {
    throw "v16 Seed51 Pilot API execution is not authorized."
}

$Repo = "D:\myx\grade_one\experiments\multi_agent_diversity"
$Python = "D:\Anaconda\envs\DL\python.exe"
$PrepRoot = Join-Path $Repo "runs\v16p51_prep"
$OutputRoot = Join-Path $Repo "runs\v16p51"
$Manifest = Join-Path $Repo "configs\task_level_comparison_strict_bbh_seed42.yaml"
$FreezePath = Join-Path $PrepRoot "source_freeze_manifest.json"
$PreregPath = Join-Path $PrepRoot "pilot_preregistration.json"
$Settings = @(
    "experimental_v16_c0_current_v15",
    "experimental_v16_c2_boundary_plus_preservation",
    "experimental_v16_c3_coalition_aware_preservation"
) -join ","

Set-Location -LiteralPath $Repo
if (-not (Test-Path -LiteralPath $FreezePath)) { throw "Source freeze manifest missing." }
if (-not (Test-Path -LiteralPath $PreregPath)) { throw "Pilot preregistration missing." }
$Freeze = Get-Content -LiteralPath $FreezePath -Raw | ConvertFrom-Json
$Prereg = Get-Content -LiteralPath $PreregPath -Raw | ConvertFrom-Json
if ((git rev-parse HEAD).Trim() -ne $Freeze.git_head) { throw "Frozen execution commit mismatch." }
if ((git status --porcelain=v1 | Out-String).Trim().Length -ne 0) { throw "Tracked worktree is dirty." }
if ($Freeze.source_freeze_status -ne "PASS") { throw "Source freeze is not PASS." }
if ($Prereg.seed -ne 51 -or $Prereg.final_test_enabled -or $Prereg.validation_enabled) {
    throw "Pilot preregistration lifecycle mismatch."
}
if (Test-Path -LiteralPath $OutputRoot) { throw "Canonical Seed51 output root must be fresh." }

$Common = @(
    "--workspace", ".", "--manifest", $Manifest,
    "--tasks", "disambiguation_qa", "--settings", $Settings,
    "--seeds", "51", "--dataset_format", "mars", "--out_root", $OutputRoot,
    "--train_size", "75", "--val_size", "50", "--test_size", "125",
    "--agent_model", "qwen3-14b", "--optimizer_model", "qwen3-14b", "--evaluator_model", "qwen3-14b",
    "--temperature", "0", "--solver_max_tokens", "1800",
    "--allow_legacy_setting", "0", "--allow_auxiliary_setting", "0",
    "--agents", "5", "--epochs", "1", "--update_every", "10",
    "--proposal_memory_mode", "off", "--num_candidates_per_parent", "2",
    "--candidate_eval_pool_size", "75", "--eval_solver_call_concurrency", "8",
    "--stage_b_candidate_budget", "2", "--resume_from_checkpoint", "0",
    "--final_test_enabled", "0"
)

& $Python scripts\preflight_member_aware.py --allow_dirty 0 @Common
if ($LASTEXITCODE -ne 0) { throw "Run-specific preflight failed before API." }
& $Python scripts\run_task_level_accuracy.py --resume_completed 0 @Common
if ($LASTEXITCODE -ne 0) { throw "Seed51 mechanism Pilot failed." }
