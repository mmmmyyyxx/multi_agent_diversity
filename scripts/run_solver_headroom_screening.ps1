$ErrorActionPreference = "Stop"
if ($env:SOLVER_HEADROOM_SCREENING_AUTHORIZED -ne "1") {
    throw "Solver Headroom Screening API execution is not authorized."
}

$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = "D:\Anaconda\envs\DL\python.exe"
$RunRoot = Join-Path $Repo "runs\solver_headroom_screening_20260901"
$FreezePath = Join-Path $RunRoot "freeze\source_freeze.json"
$PhaseAPath = Join-Path $RunRoot "phase_a\availability_smoke_private.json"
$Manifest = Join-Path $Repo "configs\task_level_comparison_strict_bbh_seed42.yaml"
Set-Location -LiteralPath $Repo

$Freeze = Get-Content -LiteralPath $FreezePath -Raw | ConvertFrom-Json
$PhaseA = Get-Content -LiteralPath $PhaseAPath -Raw | ConvertFrom-Json
if ($Freeze.source_freeze_status -ne "PASS" -or $PhaseA.gate -ne "PASS") {
    throw "Freeze and Phase A must pass."
}
if ((git rev-parse HEAD).Trim() -ne $Freeze.execution_commit) {
    throw "Execution commit mismatch."
}
if ((git status --porcelain=v1 --untracked-files=all | Out-String).Trim()) {
    throw "Worktree must be fully clean."
}
if (Test-Path -LiteralPath (Join-Path $RunRoot "train_gate.json")) {
    throw "Training already completed."
}

$Entrants = @($PhaseA.candidates | Where-Object { $_.screening_eligible })
if ($Entrants.Count -lt 1) { throw "No eligible Solver." }
$Common = @(
    "--workspace", ".", "--manifest", $Manifest,
    "--tasks", "disambiguation_qa", "--dataset_format", "mars",
    "--settings", "shared_static_reference,shared_generic_evolution",
    "--train_size", "75", "--val_size", "50", "--test_size", "125",
    "--optimizer_model", "qwen3.7-flash", "--evaluator_model", "qwen3.7-flash",
    "--temperature", "0", "--solver_max_tokens", "1800",
    "--allow_legacy_setting", "0", "--allow_auxiliary_setting", "0",
    "--agents", "5", "--epochs", "4", "--update_every", "10",
    "--proposal_memory_mode", "off", "--num_candidates_per_parent", "2",
    "--candidate_eval_pool_size", "75", "--eval_solver_call_concurrency", "8",
    "--stage_b_candidate_budget", "2", "--resume_from_checkpoint", "0",
    "--provider_call_budget", "10000", "--total_token_budget", "4000000",
    "--final_test_enabled", "0", "--preserve_final_checkpoint", "1"
)

foreach ($Seed in @(65, 66, 67)) {
    $Offset = ($Seed - 65) % $Entrants.Count
    $Ordered = @()
    for ($Index = 0; $Index -lt $Entrants.Count; $Index++) {
        $Ordered += $Entrants[($Index + $Offset) % $Entrants.Count]
    }
    foreach ($Entry in $Ordered) {
        $ModelKey = [string]$Entry.model_key
        $SolverModel = [string]$Entry.solver_model
        $SeedRoot = Join-Path $RunRoot ("training\model_" + $ModelKey + "\seed" + $Seed)
        if (Test-Path -LiteralPath $SeedRoot) { throw "Fresh seed root required." }
        & $Python scripts\preflight_member_aware.py --allow_dirty 0 @Common `
            --out_root $SeedRoot --seeds $Seed --agent_model $SolverModel
        if ($LASTEXITCODE -ne 0) { throw "Preflight failed: $ModelKey/$Seed" }
        & $Python scripts\run_task_level_accuracy.py @Common `
            --out_root $SeedRoot --seeds $Seed --agent_model $SolverModel `
            --resume_completed 0 --optimized_only 0 --immutable_comparison_cache 1
        if ($LASTEXITCODE -ne 0) { throw "Training failed: $ModelKey/$Seed" }
    }
}

& $Python scripts\audit_solver_headroom_screening.py --phase train `
    --freeze $FreezePath --out (Join-Path $RunRoot "train_gate.json")
if ($LASTEXITCODE -ne 0) { throw "Training gate failed." }
