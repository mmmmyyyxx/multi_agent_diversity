$ErrorActionPreference = "Stop"
if ($env:V17_FORMAL_VALIDATION_AUTHORIZED -ne "1") { throw "V17 validation API execution is not authorized." }
if ($env:V17_FORMAL_TRAIN_AUTHORIZED -eq "1" -or $env:V17_FORMAL_TEST_AUTHORIZED -eq "1") { throw "Validation launcher accepts only validation authorization." }
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = "D:\Anaconda\envs\DL\python.exe"
$RunRoot = Join-Path $Repo "runs\v17_formal_5arm_3seed_20260813"
$FreezePath = Join-Path $RunRoot "phase_a\source_freeze_manifest.json"
Set-Location -LiteralPath $Repo
$env:V17_FORMAL_SOURCE_FREEZE = $FreezePath
$env:PYTHONPATH = $PSScriptRoot
& $Python scripts\evaluate_v17_formal_final_states.py --phase validation --run_root $RunRoot --freeze $FreezePath
if ($LASTEXITCODE -ne 0) { throw "V17 validation failed." }
& $Python scripts\verify_v17_formal_source.py --freeze $FreezePath
if ($LASTEXITCODE -ne 0) { throw "V17 source changed during validation." }
& $Python scripts\audit_v17_formal.py --phase validation --run_root $RunRoot --freeze $FreezePath --out (Join-Path $RunRoot "validation_gate.json")
if ($LASTEXITCODE -ne 0) { throw "V17 validation gate failed." }
& $Python scripts\seal_v17_formal_test.py --run_root $RunRoot --freeze $FreezePath --out (Join-Path $RunRoot "pre_test_seal.json")
if ($LASTEXITCODE -ne 0) { throw "V17 pre-test seal failed." }
