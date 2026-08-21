param(
  [Parameter(Mandatory=$true)][string]$Registry,
  [Parameter(Mandatory=$true)][string]$SourceFreeze,
  [Parameter(Mandatory=$true)][string]$OutRoot
)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = "D:\Anaconda\envs\DL\python.exe"
if ((Resolve-Path $ProjectRoot).Path -ne (Get-Location).Path) {
  throw "Run from repository root"
}
if (Test-Path $OutRoot) { throw "Phase B output root must be fresh" }
if ($env:V17_HYBRID_TARGET_LOW_API_AUTHORIZED -ne "1") {
  throw "Explicit Phase B authorization environment latch is missing"
}
if (-not $env:DASHSCOPE_API_KEY) {
  $env:DASHSCOPE_API_KEY = [Environment]::GetEnvironmentVariable(
    "DASHSCOPE_API_KEY", "User"
  )
  if (-not $env:DASHSCOPE_API_KEY) {
    $env:DASHSCOPE_API_KEY = [Environment]::GetEnvironmentVariable(
      "DASHSCOPE_API_KEY", "Machine"
    )
  }
}
if (-not $env:DASHSCOPE_BASE_URL) {
  $env:DASHSCOPE_BASE_URL = [Environment]::GetEnvironmentVariable(
    "DASHSCOPE_BASE_URL", "User"
  )
  if (-not $env:DASHSCOPE_BASE_URL) {
    $env:DASHSCOPE_BASE_URL = [Environment]::GetEnvironmentVariable(
      "DASHSCOPE_BASE_URL", "Machine"
    )
  }
}
if (-not $env:DASHSCOPE_API_KEY) { throw "DASHSCOPE_API_KEY is unavailable" }
if (-not $env:DASHSCOPE_BASE_URL) { throw "DASHSCOPE_BASE_URL is unavailable" }
& $Python scripts\run_v17_hybrid_target_allocation_pilot.py `
  --registry $Registry --source_freeze $SourceFreeze --out_root $OutRoot
if ($LASTEXITCODE -ne 0) { throw "Pilot failed" }
& $Python scripts\audit_v17_hybrid_target_allocation_pilot.py `
  --root $OutRoot --registry $Registry --out "$OutRoot\audit.json"
if ($LASTEXITCODE -ne 0) { throw "Pilot audit failed" }
