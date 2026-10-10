"""Closed current deployment routes; credentials are never run identity."""
from .search.schemas import SearchContractError

IDENTITY = 'OPENLUX_SOLVER_LWJ_OPTIMIZER_V1'
SOLVER_MODEL = 'gpt-4o-mini'
OPTIMIZER_MODEL = 'qwen3.7-flash'


def routing_contract():
    routes = {'solver': dict(provider_profile='openlux', model=SOLVER_MODEL,
        api_key_env='OPENLUX_API_KEY', base_url_env='OPENLUX_BASE_URL')}
    for role in ('reflection', 'pattern_gradient', 'pattern_cluster'):
        routes[role] = dict(provider_profile='lwj', model=OPTIMIZER_MODEL,
            api_key_env='LWJ_DASHSCOPE_API_KEY', base_url_env='LWJ_DASHSCOPE_BASE_URL')
    return dict(identity=IDENTITY, routes=routes, unknown_role='REJECT', provider_fallback=False)


def frozen_routing(contract):
    expected = routing_contract()
    if (contract.get('provider') != IDENTITY or contract.get('provider_routing_policy') != expected
            or contract.get('models') != dict(solver=SOLVER_MODEL,
                optimizer_reflection=OPTIMIZER_MODEL, pattern=OPTIMIZER_MODEL, solver_thinking=False)):
        raise SearchContractError('CURRENT_PROVIDER_ROUTING_MISMATCH')
    return expected


def route_for_role(contract, role):
    routes = frozen_routing(contract)['routes']
    if role not in routes:
        raise SearchContractError('CURRENT_PROVIDER_ROLE_FORBIDDEN')
    return routes[role]


class ProviderClients:
    """One owner closes every independently routed client, including partial startup."""
    def __init__(self):
        self.clients = {}

    def close(self):
        seen = set()
        errors = []
        for client in self.clients.values():
            if id(client) in seen:
                continue
            seen.add(id(client))
            try:
                client.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[0]
