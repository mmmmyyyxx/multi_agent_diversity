"""Zero-API fresh V2.3 scope preparation. Domain rules live in production."""
from pathlib import Path
import argparse,json
from multi_dataset_diverse_rl.benchmarks.math_evidence_binding import prepare_v23_only_inputs

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--user-task-sha256',required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    c=prepare_v23_only_inputs(root,user_task_sha256=args.user_task_sha256)
    print(json.dumps(dict(attempt_id=c['execution_attempt_id'],binding_path=c['binding_path'],
        real_api_authorized=False,validation_calls=0,test_calls=0)))

if __name__=='__main__':main()
