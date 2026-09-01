$ErrorActionPreference = "Stop"
if ($env:MODEL_HEADROOM_SCREENING_AUTHORIZED -ne "1") {
    throw "Task Model Headroom Screening API execution is not authorized."
}

$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = "D:\Anaconda\envs\DL\python.exe"
$RunRoot = Join-Path $Repo "runs\model_headroom_screening_20260901"
$FreezePath = Join-Path $RunRoot "freeze\source_freeze.json"
$Manifest = Join-Path $Repo "configs\task_level_comparison_strict_bbh_seed42.yaml"
Set-Location -LiteralPath $Repo

$Freeze = Get-Content -LiteralPath $FreezePath -Raw | ConvertFrom-Json
if ($Freeze.source_freeze_status -ne "PASS") { throw "Source freeze is not PASS." }
if ((git rev-parse HEAD).Trim() -ne $Freeze.execution_commit) { throw "Execution commit mismatch." }
if ((git status --porcelain=v1 --untracked-files=all | Out-String).Trim()) { throw "Worktree must be fully clean." }
if (Test-Path -LiteralPath (Join-Path $RunRoot "train_gate.json")) { throw "Training already completed." }

$Models = [ordered]@{
    "A" = "qwen2.5-7b-instruct"
    "B" = "qwen3-8b"
}
$ModelOrderBySeed = @{
    62 = @("A", "B")
    63 = @("B", "A")
    64 = @("A", "B")
}
$Common = @(
    "--workspace", ".", "--manifest", $Manifest,
    "--tasks", "disambiguation_qa", "--dataset_format", "mars",
    "--settings", "shared_static_reference,shared_generic_evolution",
    "--train_size", "75", "--val_size", "50", "--test_size", "125",
    "--optimizer_model", "qwen3-14b", "--evaluator_model", "qwen3-14b",
    "--temperature", "0", "--solver_max_tokens", "1800",
    "--allow_legacy_setting", "0", "--allow_auxiliary_setting", "0",
    "--agents", "5", "--epochs", "4", "--update_every", "10",
    "--proposal_memory_mode", "off", "--num_candidates_per_parent", "2",
    "--candidate_eval_pool_size", "75", "--eval_solver_call_concurrency", "8",
    "--stage_b_candidate_budget", "2", "--resume_from_checkpoint", "0",
    "--provider_call_budget", "10000", "--total_token_budget", "4000000",
    "--final_test_enabled", "0", "--preserve_final_checkpoint", "1"
)

foreach ($Seed in @(62, 63, 64)) {
    foreach ($ModelKey in $ModelOrderBySeed[$Seed]) {
        $SeedRoot = Join-Path $RunRoot ("model_" + $ModelKey + "\seed" + $Seed)
        if (Test-Path -LiteralPath $SeedRoot) { throw "Fresh seed root required: $ModelKey/$Seed" }
        & $Python scripts\preflight_member_aware.py --allow_dirty 0 @Common `
            --out_root $SeedRoot --seeds $Seed --agent_model $Models[$ModelKey]
        if ($LASTEXITCODE -ne 0) { throw "Preflight failed: $ModelKey/$Seed" }
        & $Python scripts\run_task_level_accuracy.py @Common `
            --out_root $SeedRoot --seeds $Seed --agent_model $Models[$ModelKey] `
            --resume_completed 0 --optimized_only 0 --immutable_comparison_cache 1
        if ($LASTEXITCODE -ne 0) { throw "Training failed: $ModelKey/$Seed" }
    }
}

& $Python scripts\audit_model_headroom_screening.py --phase train `
    --freeze $FreezePath --out (Join-Path $RunRoot "train_gate.json")
if ($LASTEXITCODE -ne 0) { throw "Training gate failed." }
