param([ValidateSet("static", "generic")][string]$Phase)
$ErrorActionPreference = "Stop"
if ($env:SOLVER_MULTIMODEL_SCREENING_AUTHORIZED -ne "1") { throw "API execution not authorized" }
$Repo=(Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python="D:\Anaconda\envs\DL\python.exe"
$Root=Join-Path $Repo "runs\solver_headroom_multimodel_seed65_20260901"
$Freeze=Get-Content (Join-Path $Root "freeze\source_freeze.json") -Raw|ConvertFrom-Json
$Smoke=Get-Content (Join-Path $Root "phase_a\availability_smoke_private.json") -Raw|ConvertFrom-Json
if((git rev-parse HEAD).Trim() -ne $Freeze.execution_commit){throw "commit mismatch"}
if((git status --porcelain=v1 --untracked-files=all|Out-String).Trim()){throw "dirty worktree"}
$Manifest=Join-Path $Repo "configs\task_level_comparison_strict_bbh_seed42.yaml"
$Common=@("--workspace",".","--manifest",$Manifest,"--tasks","disambiguation_qa","--dataset_format","mars","--train_size","75","--val_size","50","--test_size","125","--optimizer_model","qwen3.7-flash","--evaluator_model","qwen3.7-flash","--temperature","0","--solver_max_tokens","1800","--allow_legacy_setting","0","--allow_auxiliary_setting","0","--agents","5","--epochs","4","--update_every","10","--proposal_memory_mode","off","--num_candidates_per_parent","2","--candidate_eval_pool_size","75","--eval_solver_call_concurrency","8","--stage_b_candidate_budget","2","--provider_call_budget","10000","--total_token_budget","4000000","--final_test_enabled","0","--preserve_final_checkpoint","1")
if($Phase -eq "static"){$Entries=@($Smoke.candidates|Where-Object{$_.static_eligible -and $_.key -ne "Q8"})}else{$Selection=Get-Content (Join-Path $Root "static_selection_private.json") -Raw|ConvertFrom-Json;$Entries=@($Selection.selected|Where-Object{$_.key -ne "Q8"})}
foreach($Entry in $Entries){
  $Out=Join-Path $Root ("training\model_"+$Entry.key+"\seed65")
  if($Phase -eq "static"){
    if(Test-Path $Out){throw "fresh static root required"}
    $Settings="shared_static_reference";$Resume="0"
  }else{
    if(-not (Test-Path $Out)){throw "missing static root"}
    $Settings="shared_static_reference,shared_generic_evolution";$Resume="1"
  }
  & $Python scripts\preflight_member_aware.py --allow_dirty 0 @Common --settings $Settings --out_root $Out --seeds 65 --agent_model $Entry.model
  if($LASTEXITCODE -ne 0){throw "preflight failed $($Entry.key)"}
  & $Python scripts\run_task_level_accuracy.py @Common --settings $Settings --out_root $Out --seeds 65 --agent_model $Entry.model --resume_completed $Resume --optimized_only 0 --immutable_comparison_cache 1
  if($LASTEXITCODE -ne 0){throw "run failed $($Entry.key)"}
}
