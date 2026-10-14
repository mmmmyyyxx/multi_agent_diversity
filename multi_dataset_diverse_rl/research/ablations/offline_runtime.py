"""Guarded fake transport only; reuse production scoring/accounting primitives.

This adapter is deliberately incapable of authorizing real execution. A future
attempt needs a governed phase/source/data freeze and fresh execution binding.
"""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
import socket

from ...benchmarks.data_freeze import digest
from ...search.schemas import SearchContractError
from ..single_prompt_runtime import StudyBroker
from ..gated_prompt_network import NetworkRecoveryMixin
from .config import validate_plan, request_namespace


class OfflineAblationBroker(NetworkRecoveryMixin, StudyBroker):
    def __init__(self, *, plan, **kwargs):
        # sitecustomize must precede every provider import in the launching process.
        if socket.socket.connect.__module__ != 'sitecustomize':
            raise SearchContractError('ABLATION_OFFLINE_NETWORK_GUARD_REQUIRED')
        validate_plan(plan)
        self.plan = deepcopy(plan)
        self.repeat_id = None
        super().__init__(**kwargs)

    def study_policy(self):
        return deepcopy(self.plan['config']['runtime'])

    def timeout_policy(self):
        return self.study_policy()['timeout_recovery']

    def bind_cell(self, cell, plan):
        if plan != self.plan or cell not in plan['cells'] or self.phase != 'train':
            raise SearchContractError('ABLATION_BROKER_CELL_IDENTITY')
        self.cell_id = cell['cell_id']
        self.attempt_id = 'SYNTHETIC_' + cell['cell_id']
        self.seed, self.repeat_id = cell['seed'], cell['repeat_id']
        self.round_index = 0

    def legal_role_split(self, role, split):
        return ((self.phase == 'train' and ((split == 'optimize' and role in
                 ('solver', 'reflection', 'pattern_gradient', 'pattern_cluster'))
                 or (split == 'validation' and role == 'solver')))
                or (self.phase == 'test' and split == 'test' and role == 'solver'))

    def cache_ordinal(self, split):
        return 0 if split == 'validation' else self.round_index

    def _request_identity(self, **kwargs):
        if self.repeat_id is None:
            raise SearchContractError('ABLATION_REQUEST_REPEAT_NOT_BOUND')
        request, base = super()._request_identity(**kwargs)
        return request, digest([base, request_namespace(self.plan['plan_identity'], self.seed, self.repeat_id)])

    def evaluate(self, prompt, rows, split, seed, ordinal):
        if not rows or seed != self.seed or not self.legal_role_split('solver', split):
            raise SearchContractError('ABLATION_EVALUATION_SCOPE')
        self.round_index = ordinal
        workers = self.study_policy()['test_solver_concurrency' if split == 'test' else 'solver_concurrency']
        rows = tuple(rows)
        values = [None] * len(rows)
        cursor = 0
        failure = None
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='ablation_solver') as pool:
            pending = {}
            while cursor < len(rows) or pending:
                while failure is None and cursor < len(rows) and len(pending) < workers:
                    pending[pool.submit(self.prediction, prompt, rows[cursor], split, seed, ordinal)] = cursor
                    cursor += 1
                if not pending:
                    break
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in sorted(completed, key=pending.get):
                    index = pending.pop(future)
                    try:
                        values[index] = future.result()
                    except BaseException as exc:
                        if failure is None:
                            failure = exc
            if failure is not None:
                raise failure
        return {r['stable_example_id']: v for r, v in zip(rows, values)}
