param([ValidateSet("generic")][string]$Phase="generic")
$ErrorActionPreference = "Stop"
if ($env:SOLVER_MULTIMODEL_SCREENING_AUTHORIZED -ne "1") { throw "API execution not authorized" }
$Repo=(Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python="D:\Anaconda\envs\DL\python.exe"
$Root=Join-Path $Repo "runs\solver_headroom_multimodel_seed65_20260901"
$Freeze=Get-Content (Join-Path $Root "generic_retry2_freeze\source_freeze.json") -Raw|ConvertFrom-Json
$Smoke=Get-Content (Join-Path $Root "phase_a\availability_smoke_private.json") -Raw|ConvertFrom-Json
if((git rev-parse HEAD).Trim() -ne $Freeze.execution_commit){throw "commit mismatch"}
if((git status --porcelain=v1 --untracked-files=all|Out-String).Trim()){throw "dirty worktree"}
$Manifest=Join-Path $Repo "configs\task_level_comparison_strict_bbh_seed42.yaml"
$Common=@("--workspace",".","--manifest",$Manifest,"--tasks","disambiguation_qa","--dataset_format","mars","--train_size","75","--val_size","50","--test_size","125","--optimizer_model","qwen3.7-flash","--evaluator_model","qwen3.7-flash","--temperature","0","--solver_max_tokens","1800","--allow_legacy_setting","0","--allow_auxiliary_setting","0","--agents","5","--epochs","4","--update_every","10","--proposal_memory_mode","off","--num_candidates_per_parent","2","--candidate_eval_pool_size","75","--eval_solver_call_concurrency","8","--stage_b_candidate_budget","2","--provider_call_budget","10000","--total_token_budget","4000000","--final_test_enabled","0","--preserve_final_checkpoint","1")
$Selection=Get-Content (Join-Path $Root "static_selection_retry2_private.json") -Raw|ConvertFrom-Json
$Entries=@($Selection.selected|Where-Object{$_.key -ne "Q8"})
foreach($Entry in $Entries){
  $StaticRun=Join-Path $Root ("training\model_"+$Entry.key+"\seed65\disambiguation_qa\shared_static_reference_seed65")
  $StaticCache=Join-Path $StaticRun "_solver_cache.sqlite"
  if(-not (Test-Path $StaticCache)){throw "missing completed Static cache $($Entry.key)"}
  $Out=Join-Path $Root ("generic_retry2\training\model_"+$Entry.key+"\seed65")
  $CacheDir=Join-Path $Root "generic_retry2\cache"
  $Cache=Join-Path $CacheDir ("model_"+$Entry.key+".sqlite")
  if((Test-Path $Out) -or (Test-Path $Cache)){throw "fresh Generic retry root required"}
  New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null
  Copy-Item -LiteralPath $StaticCache -Destination $Cache
  $Settings="shared_generic_evolution"
  & $Python scripts\preflight_member_aware.py --allow_dirty 0 @Common --settings $Settings --out_root $Out --seeds 65 --agent_model $Entry.model --shared_solver_cache_path ([System.IO.Path]::GetFullPath($Cache))
  if($LASTEXITCODE -ne 0){throw "preflight failed $($Entry.key)"}
  & $Python scripts\run_task_level_accuracy.py @Common --settings $Settings --out_root $Out --seeds 65 --agent_model $Entry.model --shared_solver_cache_path ([System.IO.Path]::GetFullPath($Cache)) --resume_completed 0 --optimized_only 1 --immutable_comparison_cache 1
  if($LASTEXITCODE -ne 0){throw "run failed $($Entry.key)"}
}
