param(
    [string]$OutRoot = "runs_v4_full_highfreq_seed42_$(Get-Date -Format yyyyMMdd_HHmmss)"
)

$ErrorActionPreference = "Stop"
$PY = "D:\Anaconda\envs\DL\python.exe"

if (-not $env:DASHSCOPE_API_KEY) { throw "DASHSCOPE_API_KEY is not set" }
if (Test-Path -LiteralPath $OutRoot) { throw "out_root must be fresh" }

& $PY scripts\run_v4_full_high_frequency_seed42.py --workspace . --out_root $OutRoot
if ($LASTEXITCODE -ne 0) { throw "high-frequency Full pilot failed" }
