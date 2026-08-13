$ErrorActionPreference = "Stop"
if ($env:V17_FORMAL_TEST_AUTHORIZED -ne "1") { throw "V17 test API execution is not authorized." }
if ($env:V17_FORMAL_TRAIN_AUTHORIZED -eq "1" -or $env:V17_FORMAL_VALIDATION_AUTHORIZED -eq "1") { throw "Test launcher accepts only test authorization." }
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = "D:\Anaconda\envs\DL\python.exe"
$RunRoot = Join-Path $Repo "runs\v17_formal_5arm_3seed_20260813"
$FreezePath = Join-Path $RunRoot "phase_a\source_freeze_manifest.json"
Set-Location -LiteralPath $Repo
$env:V17_FORMAL_SOURCE_FREEZE = $FreezePath
$env:PYTHONPATH = $PSScriptRoot
& $Python scripts\evaluate_v17_formal_final_states.py --phase test --run_root $RunRoot --freeze $FreezePath
if ($LASTEXITCODE -ne 0) { throw "V17 test failed." }
& $Python scripts\verify_v17_formal_source.py --freeze $FreezePath
if ($LASTEXITCODE -ne 0) { throw "V17 source changed during test." }
& $Python scripts\audit_v17_formal.py --phase test --run_root $RunRoot --freeze $FreezePath --out (Join-Path $RunRoot "test_gate.json")
if ($LASTEXITCODE -ne 0) { throw "V17 test gate failed." }
