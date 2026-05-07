if (-not $env:OPENAI_API_KEY) {
  throw "OPENAI_API_KEY is not set in environment. Please run: `$env:OPENAI_API_KEY='your_key'"
}

$TaskType = if ($args.Count -gt 0) { $args[0] } else { "auto" }
$TrainPath = if ($args.Count -gt 1) { $args[1] } else { "train.jsonl" }
$TestPath = if ($args.Count -gt 2) { $args[2] } else { "test.jsonl" }

if ($TaskType -ne "auto" -and $TaskType -ne "gsm8k" -and $TaskType -ne "mmlu") {
  Write-Host "ERROR: TASK_TYPE must be one of: auto, gsm8k, mmlu"
  Write-Host "Usage: powershell -File run.ps1 [task_type] [train_path] [test_path]"
  exit 1
}

python -m multi_dataset_diverse_rl.cli `
  --task_type "$TaskType" `
  --model gpt-4o-mini `
  --critic_model gpt-4o-mini `
  --rewriter_model gpt-4o-mini `
  --max_retries 5 `
  --retry_sleep 2.0 `
  --train_path "$TrainPath" `
  --test_path "$TestPath" `
  --agents 4 `
  --epochs 2 `
  --update_every 5 `
  --train_size 200 `
  --test_size 100 `
  --candidate_eval_batch_size 3
