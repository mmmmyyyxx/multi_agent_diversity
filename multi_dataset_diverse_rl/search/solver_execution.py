"""Frozen Solver execution physics; optimization decisions remain serial."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy

from .schemas import SearchContractError


POLICY = dict(identity='STRUCTURED_SOLVER_CAPACITY_EXECUTION_V2',
    solver_max_concurrency=8, optimizer_max_concurrency=1,
    first_output_tokens=3600, length_recovery_output_tokens=6144,
    expansion_trigger='IMMEDIATELY_PREVIOUS_RESPONSE_LENGTH_TRUNCATED',
    max_semantic_attempts=4, result_order='INPUT_ORDER',
    inflight_identical_key='SINGLE_OWNER_SHARED_FUTURE',
    batch_failure='STOP_DISPATCH_DRAIN_PERSIST_THEN_RAISE',
    optimization_dependencies='SERIAL')


def frozen_execution_policy(contract):
    policy = contract.get('solver_execution_policy')
    # Existing immutable freezes retain their original serial execution physics.
    if policy is None:
        return None
    if policy != POLICY or any(type(policy[k]) is not type(v) for k, v in POLICY.items()):
        raise SearchContractError('SOLVER_EXECUTION_POLICY_MISMATCH')
    return deepcopy(policy)


def next_capacity(policy, previous=None):
    if policy is None:
        return 3600
    return (policy['length_recovery_output_tokens'] if previous is not None
        and previous.finish_reason in {'length', 'max_tokens', 'max_output_tokens'}
        else policy['first_output_tokens'])


def bounded_ordered_map(function, items, concurrency):
    """Bound queued work too; on failure drain already issued requests before return.

    No caller may close its ledger while a worker can still receive a response.
    This synchronous pool also serves synchronous initialization ports.
    """
    items = tuple(items)
    if type(concurrency) is not int or not 1 <= concurrency <= 8:
        raise SearchContractError('SOLVER_BATCH_CONCURRENCY_INVALID')
    if concurrency == 1:
        return tuple(function(item) for item in items)
    results = [None] * len(items)
    cursor = 0
    failure = None
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix='solver') as pool:
        pending = {}
        while cursor < len(items) or pending:
            while failure is None and cursor < len(items) and len(pending) < concurrency:
                pending[pool.submit(function, items[cursor])] = cursor
                cursor += 1
            if not pending:
                break
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in sorted(completed, key=pending.get):
                index = pending.pop(future)
                try:
                    results[index] = future.result()
                except BaseException as exc:
                    if failure is None:
                        failure = exc
            # Do not replenish until every completion in this wave was checked.
        if failure is not None:
            raise failure
    return tuple(results)
