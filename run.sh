if [ -z "${OPENAI_API_KEY}" ]; then
  echo "ERROR: OPENAI_API_KEY is not set in environment."
  echo "Please run: export OPENAI_API_KEY=your_key"
  exit 1
fi

TASK_TYPE="${1:-auto}"
TRAIN_PATH="${2:-train.jsonl}"
TEST_PATH="${3:-test.jsonl}"

if [ "$TASK_TYPE" != "auto" ] && [ "$TASK_TYPE" != "gsm8k" ] && [ "$TASK_TYPE" != "mmlu" ]; then
  echo "ERROR: TASK_TYPE must be one of: auto, gsm8k, mmlu"
  echo "Usage: bash run.sh [task_type] [train_path] [test_path]"
  exit 1
fi

python -m multi_dataset_diverse_rl.cli \
  --task_type "$TASK_TYPE" \
  --model gpt-4o-mini \
  --critic_model gpt-4o-mini \
  --rewriter_model gpt-4o-mini \
  --max_retries 5 \
  --retry_sleep 2.0 \
  --train_path "$TRAIN_PATH" \
  --test_path "$TEST_PATH" \
  --agents 4 \
  --epochs 2 \
  --update_every 5 \
  --train_size 200 \
  --test_size 100 \
  --candidate_eval_batch_size 3