[CmdletBinding()]
param(
    [string]$OutRoot = "runs_matched_gpt4omini_seed42_20260725"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$python = "D:\Anaconda\envs\DL\python.exe"
$manifest = "configs/task_level_comparison_strict_bbh_seed42.yaml"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "DL Conda interpreter not found: $python"
}

if ([IO.Path]::IsPathRooted($OutRoot)) {
    $outPath = [IO.Path]::GetFullPath($OutRoot)
} else {
    $outPath = [IO.Path]::GetFullPath((Join-Path $repoRoot $OutRoot))
}
$expectedParent = $repoRoot.TrimEnd([IO.Path]::DirectorySeparatorChar)
$actualParent = [IO.Path]::GetDirectoryName($outPath).TrimEnd([IO.Path]::DirectorySeparatorChar)
$outName = [IO.Path]::GetFileName($outPath)
if (-not $actualParent.Equals($expectedParent, [StringComparison]::OrdinalIgnoreCase)) {
    throw "OutRoot must be a direct child of the repository: $outPath"
}
if ($outName -notmatch '^runs_matched_gpt4omini_seed42_[0-9]{8}(?:_[0-9]{6})?$') {
    throw "OutRoot must use a fresh matched-pilot name: $outName"
}
if (Test-Path -LiteralPath $outPath) {
    throw "OutRoot already exists; choose a new empty output directory: $outPath"
}

$experimentArgs = @(
    "--manifest", $manifest,
    "--tasks", "disambiguation_qa",
    "--settings", "shared_baseline,shared_independent_accuracy,shared_member_aware_full",
    "--seeds", "42",
    "--dataset_format", "mars",
    "--out_root", $outName,
    "--agent_model", "gpt-4o-mini",
    "--optimizer_model", "gpt-4o-mini",
    "--evaluator_model", "gpt-4o-mini",
    "--agents", "5",
    "--initialization_mode", "shared_identical",
    "--train_size", "75",
    "--val_size", "50",
    "--test_size", "125",
    "--epochs", "8",
    "--update_every", "75",
    "--candidate_eval_pool_size", "75",
    "--num_candidates_per_parent", "2",
    "--stage_a_representative_size", "12",
    "--stage_a_coverage_size", "6",
    "--stage_a_conversion_size", "6",
    "--stage_a_preservation_size", "4",
    "--stage_a_channel_top_k", "2",
    "--stage_b_candidate_budget", "2",
    "--solver_max_tokens", "1800",
    "--solver_invalid_max_retries", "3",
    "--eval_solver_call_concurrency", "8",
    "--resume_from_checkpoint", "0"
)

Push-Location $repoRoot
try {
    Write-Host "Running matched-pilot preflight (no API calls)..."
    & $python "scripts\preflight_member_aware.py" `
        "--workspace" "." `
        "--allow_dirty" "0" `
        @experimentArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Matched-pilot preflight failed with exit code $LASTEXITCODE"
    }

    Write-Host "Starting GPT-4o-mini matched efficacy pilot: $outName"
    & $python "scripts\run_task_level_accuracy.py" `
        "--workspace" "." `
        "--resume_completed" "0" `
        @experimentArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Matched efficacy pilot failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
