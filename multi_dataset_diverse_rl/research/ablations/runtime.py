"""Repeat-scoped Solver observations; other roles always dispatch fresh requests.

Provider construction and authorization belong exclusively to governed execution.
"""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
import threading

from ...benchmarks.data_freeze import digest
from ...search.schemas import SearchContractError
from ..single_prompt_runtime import StudyBroker
from ..gated_prompt_network import NetworkRecoveryMixin
from .config import validate_plan, request_namespace
from ...persistence.durable_io import append_jsonl


class AblationBroker(NetworkRecoveryMixin, StudyBroker):
    def __init__(self, *, plan, **kwargs):
        validate_plan(plan)
        self.plan = deepcopy(plan)
        self.repeat_id = None
        self.example_context = threading.local()
        super().__init__(**kwargs)

    def study_policy(self):
        return deepcopy(self.plan['config']['runtime'])

    def timeout_policy(self):
        return self.study_policy()['timeout_recovery']

    def bind_cell(self, cell, plan):
        if plan != self.plan or cell not in plan['cells'] or self.phase not in ('train', 'test'):
            raise SearchContractError('ABLATION_BROKER_CELL_IDENTITY')
        self.cell_id = cell['cell_id']
        self.attempt_id = self.scope['scope_identity'] + '_' + self.phase + '_' + cell['cell_id']
        self.seed, self.repeat_id = cell['seed'], cell['repeat_id']
        self.round_index = 0

    def legal_role_split(self, role, split):
        return ((self.phase == 'train' and ((split == 'optimize' and role in
                 ('solver', 'reflection', 'pattern_gradient', 'pattern_cluster'))
                 or (split == 'validation' and role == 'solver')))
                or (self.phase == 'test' and split == 'test' and role == 'solver'))

    def cache_ordinal(self, split):
        return 0

    def _request_identity(self, **kwargs):
        if self.repeat_id is None:
            raise SearchContractError('ABLATION_REQUEST_REPEAT_NOT_BOUND')
        request, base = super()._request_identity(**kwargs)
        identity = [base, request_namespace(self.plan['plan_identity'], self.seed, self.repeat_id)]
        if kwargs['role'] == 'solver':
            context = getattr(self.example_context, 'identity', None)
            if context is None:
                raise SearchContractError('ABLATION_SOLVER_EXAMPLE_IDENTITY_REQUIRED')
            identity.append(context)
        else:
            identity.extend([self.cell_id, self.round_index])
        return request, digest(identity)

    def complete(self, **kwargs):
        if self.replay is not None:
            raise SearchContractError('ABLATION_NON_SOLVER_REPLAY_FORBIDDEN')
        return super().complete(**kwargs)

    def prediction(self, prompt, row, split, seed, ordinal):
        if seed != self.seed or not self.legal_role_split('solver', split):
            raise SearchContractError('ABLATION_SOLVER_SCOPE')
        self.example_context.identity = dict(example_id=row['stable_example_id'],
            prompt_sha256=prompt.prompt_hash, problem_sha256=digest(row['content']['problem']),
            scoring_reference_sha256=digest(row['reference_final_answer']),
            scoring_contract='PINNED_MATH_FIRST_VALID_FOUR_DRAWS_V1')
        try:
            value = super().prediction(prompt, row, split, seed, ordinal)
            # Share the scored terminal observation, including semantic recovery.
            with self.lock:
                append_jsonl(self.receipts.directory / 'solver_cache_access.jsonl', dict(
                    cell_id=self.cell_id, repeat_id=self.repeat_id, split=split,
                    ordinal=ordinal, example_id=row['stable_example_id'],
                    prompt_sha256=prompt.prompt_hash, request_sha256=value['request_sha256'],
                    observation_sha256=digest(value)))
            return value
        finally:
            del self.example_context.identity

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
