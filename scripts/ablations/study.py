"""Thin CLI for isolated draft planning and guarded synthetic execution."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    from multi_dataset_diverse_rl.research.ablations.config import build_plan, current_config
    from multi_dataset_diverse_rl.research.ablations.formal_adapters import production_memory_binding
    from multi_dataset_diverse_rl.persistence.durable_io import atomic_write_json
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--plan', action='store_true')
    modes.add_argument('--smoke', action='store_true')
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    config = current_config() if args.config is None else json.loads(args.config.read_text(encoding='utf-8'))
    plan = build_plan(config)
    destination = args.destination.resolve()
    if not destination.is_relative_to((ROOT / 'runs').resolve()):
        raise SystemExit('Private outputs must remain under ignored runs/.')
    if args.plan:
        if destination.exists():
            raise SystemExit('Fresh destination required.')
        atomic_write_json(destination / 'plan.json', plan)
        atomic_write_json(destination / 'future_memory_binding.json', production_memory_binding())
        from multi_dataset_diverse_rl.research.ablations.semantic_alignment import inventory
        atomic_write_json(destination / 'module_alignment_inventory.json',inventory())
        print(json.dumps(dict(status='DRAFT_NOT_AUTHORIZED', cells=len(plan['cells']),
            seed=config['seed'], repeats_per_arm=config['repeats_per_arm'],
            plan_identity=plan['plan_identity'], real_api_calls=0), indent=2))
    else:
        from multi_dataset_diverse_rl.research.ablations.synthetic import smoke
        summary = smoke(plan, destination)
        print(json.dumps({k: summary[k] for k in ('status', 'groups', 'real_api_calls', 'network_attempts', 'unresolved_reservations')}, indent=2))


if __name__ == '__main__':
    main()
