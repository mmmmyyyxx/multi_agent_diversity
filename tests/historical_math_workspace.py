"""Explicit test-only V1_1 artifact workspace; no legacy runtime is modified."""
import json
from pathlib import Path
import shutil
import subprocess


def build_historical_math_workspace(root, destination):
    destination.mkdir(parents=True,exist_ok=True)
    # Copy, rather than link, tracked inputs so test mutations cannot reach
    # immutable source/evidence or change the current V1_2 initialization.
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=root).decode().split('\0')
    prefixes=('multi_dataset_diverse_rl/','experiments/','docs/','scripts/','infrastructure/','tests/fixtures/')
    for relative in tracked:
        if not relative or (not relative.startswith(prefixes) and '/' in relative):continue
        source=root/relative
        if not source.is_file():continue
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    initial='experiments/initial_teams/math_generic_team_seed_v1_1.json'
    shutil.copyfile(root/'experiments/initial_teams/archive/math_generic_team_seed_v1_1.json',destination/initial)
    assert json.loads((destination/initial).read_bytes())['team_version']=='MATH_GENERIC_TEAM_SEED_V1_1'
    for name in ('math_v2_1_low_cost_canary_v1.json','math_v2_1_canary_v1.json'):
        c=json.loads((root/'experiments/execution_bindings'/name).read_bytes())
        canonical=c['canonical_root']
        if not (destination/canonical).exists():
            for relative in ('manifests/math.json','raw/math'):
                source=root/canonical/relative;target=destination/canonical/relative
                target.parent.mkdir(parents=True,exist_ok=True)
                if source.is_dir():shutil.copytree(source,target)
                else:shutil.copyfile(source,target)
    return destination
