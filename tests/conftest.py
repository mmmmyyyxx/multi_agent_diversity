"""Explicit current/full-historical suite selection; no silent private replay."""
import json
import importlib
from pathlib import Path
import re

import pytest

ROOT=Path(__file__).resolve().parents[1]
CLASSIFICATION=json.loads((ROOT/'tests/suite_classification.json').read_text(encoding='utf-8'))
ROWS={r['path']:r for r in CLASSIFICATION['tests']}

# These compatibility tests explicitly exercise frozen older MATH treatments.
# Their V1_1 fixture must not read the new V1_2 initial-condition artifact.
HISTORICAL_MATH_FIXTURE_MODULES={
    'tests/test_math_v21_binding.py','tests/test_math_answer_domain_v2.py',
    'tests/test_math_low_cost.py','tests/test_math_optimizer_generation.py',
    'tests/test_math_layer1_binding.py',
    'tests/test_math_preexecution.py','tests/test_math_solver_decoding.py',
    'tests/test_math_prediction_validity.py','tests/test_math_output_reuse.py',
    'tests/test_math_token_accounting.py','tests/test_math_terminal_recovery.py',
}


@pytest.fixture(scope='session')
def historical_math_workspace(tmp_path_factory):
    from tests.historical_math_workspace import build_historical_math_workspace
    return build_historical_math_workspace(ROOT,tmp_path_factory.mktemp('historical_math_v1_1'))


@pytest.fixture
def tmp_path(request,tmp_path_factory):
    # Historical helpers create root-relative synthetic binding receipts. Keep
    # those writes in the isolated workspace, never in current tracked inputs.
    name=re.sub(r'\W','_',request.node.name)[:30]
    path=tmp_path_factory.mktemp(name,numbered=True)
    relative=Path(request.node.path).resolve().relative_to(ROOT).as_posix()
    if relative in HISTORICAL_MATH_FIXTURE_MODULES:
        path=request.getfixturevalue('historical_math_workspace')/'runs'/path.name
        path.mkdir(parents=True,exist_ok=False)
    return path


@pytest.fixture
def historical_math_helpers(historical_math_workspace,monkeypatch):
    # Shared helpers use one explicit historical root, including when one
    # compatibility module calls another module's fake execution function.
    for relative in sorted(HISTORICAL_MATH_FIXTURE_MODULES):
        module=importlib.import_module(relative[:-3].replace('/','.'))
        monkeypatch.setattr(module,'ROOT',historical_math_workspace)
    return historical_math_workspace


@pytest.fixture(autouse=True)
def explicit_historical_math_fixture(request,monkeypatch):
    relative=Path(request.node.path).resolve().relative_to(ROOT).as_posix()
    if relative in HISTORICAL_MATH_FIXTURE_MODULES:
        workspace=request.getfixturevalue('historical_math_helpers')
        # Pytest may collect a test file by its unqualified module name while
        # cross-module helpers import it through the tests namespace.
        monkeypatch.setattr(request.module,'ROOT',workspace)


def pytest_addoption(parser):
    parser.addoption('--suite',choices=('current','full-historical'),default='current',
                     help='current contracts or complete historical collection with explicit missing-private skips')


def pytest_ignore_collect(collection_path,config):
    """Exclude explicitly retired modules before importing their old runtime.

    Replay tests remain unchanged and require their original source checkout.
    The complete omitted-module inventory is reported, never counted as PASS.
    """
    try:relative=collection_path.resolve().relative_to(ROOT).as_posix()
    except ValueError:return None
    row=ROWS.get(relative)
    if config.getoption('--suite')=='current' and row is not None and not row['current_suite_included']:
        return True
    return None


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config,items):
    mode=config.getoption('--suite');kept=[];excluded=[];private=0;historical=0
    for item in items:
        relative=Path(item.path).resolve().relative_to(ROOT).as_posix()
        row=ROWS.get(relative)
        if row is None:
            raise pytest.UsageError('Unclassified test module: '+relative)
        category=row['classification']
        if category=='HISTORICAL_PRIVATE_ARTIFACT':
            item.add_marker(pytest.mark.historical_private_artifact)
            private+=1
            missing=[p for p in row['required_private_artifacts'] if not (ROOT/p).exists()]
            if mode=='full-historical' and missing:
                item.add_marker(pytest.mark.skip(reason='historical private-artifact assets missing: '+', '.join(missing)))
        elif category=='HISTORICAL_REPLAY':
            item.add_marker(pytest.mark.historical_replay);historical+=1
        else:
            item.add_marker(pytest.mark.current_contract)
        if mode=='current' and not row['current_suite_included']:excluded.append(item)
        else:kept.append(item)
    items[:]=kept
    if excluded:config.hook.pytest_deselected(items=excluded)
    config._repository_suite_summary=dict(mode=mode,selected=len(kept),deselected=len(excluded),
        private_artifact_test_cases=private,historical_replay_test_cases=historical,
        omitted_modules=[dict(path=r['path'],classification=r['classification'],
            original_source=r.get('replay_source'),scope_note=r.get('scope_note'))
            for r in CLASSIFICATION['tests'] if mode=='current' and not r['current_suite_included']])


def pytest_terminal_summary(terminalreporter,exitstatus,config):
    summary=getattr(config,'_repository_suite_summary',{})
    terminalreporter.write_sep('=', 'Repository suite scope')
    terminalreporter.write_line(json.dumps({**{k:v for k,v in summary.items() if k!='omitted_modules'},
        'omitted_module_count':len(summary.get('omitted_modules',[])),
        'omitted_module_inventory':'tests/suite_classification.json'},sort_keys=True))
    terminalreporter.write_line('historical private-artifact tests not executed when excluded or required assets are missing; full historical replay PASS is not claimed')
