$ErrorActionPreference = "Stop"

if ($env:V16_FIXED_PARENT_PROBE_AUTHORIZED -ne "1") {
    throw "Fixed-parent probe API execution is not authorized."
}

$Repo = "D:\myx\grade_one\experiments\multi_agent_diversity"
$Python = "D:\Anaconda\envs\DL\python.exe"
$Registry = Join-Path $Repo "runs\v16_fixed_parent_probe_prep_20260811\case_registry.json"
$OutputRoot = Join-Path $Repo "runs\v16_fixed_parent_probe_seed51"

Set-Location -LiteralPath $Repo
if ((git status --porcelain=v1 | Out-String).Trim().Length -ne 0) {
    throw "Tracked worktree must be clean."
}
if (-not (Test-Path -LiteralPath $Registry)) { throw "Frozen case registry is missing." }
if (Test-Path -LiteralPath $OutputRoot) { throw "Probe output root must be fresh." }

& $Python scripts\preflight_v16_fixed_parent_generation_probe.py --registry $Registry
if ($LASTEXITCODE -ne 0) { throw "Fixed-parent probe preflight failed." }
& $Python scripts\run_v16_fixed_parent_generation_probe.py --registry $Registry --out_root $OutputRoot
if ($LASTEXITCODE -ne 0) { throw "Fixed-parent generation probe failed." }
