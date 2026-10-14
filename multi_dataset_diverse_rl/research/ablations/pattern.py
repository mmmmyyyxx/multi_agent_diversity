"""Production discovery with the previously approved independent largest focus."""
from copy import deepcopy
from pathlib import Path
from ...search.pattern_selection import LargestUniqueSupportSelector

from ...benchmarks.data_freeze import digest, file_hash
from ...search.schemas import SearchContractError
from ...search.textual_gradients import GRADIENT_POLICY, REFERENCE_GRADIENT_PROMPT, CLUSTER_PROMPT
from ...search.generation_failures import POLICY as OUTPUT_RECOVERY
from ...current_contract import GRADIENT_PARTITION_COMPLETION_VERSION
from ..single_prompt import raw_error
from .interfaces import PreparedFeedback
from .formal_adapters import discover


def project(feedback, seed, ordinal):
    full = deepcopy(feedback)
    sf, errors = full['structured_feedback'], full['errors']
    if not isinstance(sf, dict) or set(sf) != {
            'gradients', 'diagnostics', 'patterns', 'unassigned_ids', 'partition_audit', 'cluster_structural_errors'}:
        raise SearchContractError('ABLATION_PATTERN_FEEDBACK_CHANNELS')
    batch = {e['example_id'] for e in errors}
    grad_ids = {g['example_id'] for g in sf['gradients']}
    patterns = sf['patterns']
    ids = [p['pattern_id'] for p in patterns]
    if len(batch) != len(errors) or len(grad_ids) != len(sf['gradients']) or len(set(ids)) != len(ids):
        raise SearchContractError('ABLATION_PATTERN_DUPLICATE_EVIDENCE')
    supports = {p['pattern_id']: set(p['support_ids']) for p in patterns}
    if (not grad_ids <= batch or any(not support or not support <= batch & grad_ids for support in supports.values())
            or any(d.get('example_id') not in batch for d in sf['diagnostics'])):
        raise SearchContractError('ABLATION_PATTERN_SUPPORT_MISMATCH')
    audit = dict(seed=seed, round=ordinal, pattern_sizes={p: len(v) for p, v in sorted(supports.items())},
        selected_pattern_id=None, selected_support_ids=[], tie_ids=[], fallback_to_B=not bool(patterns))
    if not patterns:
        return dict(errors=errors, structured_feedback=None), audit
    maximum = max(map(len, supports.values()))
    ties = sorted(p for p, support in supports.items() if len(support) == maximum)
    selected,ties=LargestUniqueSupportSelector().select(patterns,seed=seed,ordinal=ordinal)
    chosen=selected['pattern_id']
    support = supports[chosen]
    audit.update(selected_pattern_id=chosen, selected_support_ids=sorted(support), tie_ids=ties)
    complete = (len(patterns) == 1 and support == batch == grad_ids
        and not sf['unassigned_ids'] and not sf['cluster_structural_errors']
        and all(not row.get('discarded_patterns') for row in sf['partition_audit']))
    if not complete:
        sf = dict(sf, patterns=[selected], gradients=[g for g in sf['gradients'] if g['example_id'] in support],
            diagnostics=[d for d in sf['diagnostics'] if d['example_id'] in support],
            unassigned_ids=[], partition_audit=[], cluster_structural_errors=[])
    return dict(errors=[e for e in errors if e['example_id'] in support], structured_feedback=sf), audit


class LargestPatternModule:
    def binding(self):
        root = Path(__file__).resolve().parents[2]
        files = ['search/textual_gradients.py', 'search/pattern_primitives.py', 'search/pattern_selection.py',
                 'search/optimizer_output_contract.py', 'search/generation_failures.py',
                 'search/weighted_responsibility.py', 'search/optimization_evidence.py',
                 'search/readable_evidence.py', 'search/partition_completion.py',
                 'benchmarks/math_response_evidence.py', 'current_contract.py', 'versions.py']
        return dict(identity='ABLATION_FORMAL_DISCOVERY_LARGEST_SUPPORT_V1',
            implementation='ablations.pattern.LargestPatternModule', required_error_source='random_wrong_batch',
            semantic_calls_per_opportunity=12, maximum_state_bytes=4,
            production_implementations=['search.textual_gradients.GradientExtractor',
                'search.textual_gradients.PerExampleGradientProvider',
                'search.textual_gradients.GradientClusterProvider',
                'search.textual_gradients.normalize_gradient_partition'],
            production_source_hashes={p: file_hash(root / p) for p in files},
            gradient_policy=deepcopy(GRADIENT_POLICY), output_recovery=deepcopy(OUTPUT_RECOVERY),
            partition_completion_policy=GRADIENT_PARTITION_COMPLETION_VERSION,
            gradient_prompt_sha256=digest(REFERENCE_GRADIENT_PROMPT), cluster_prompt_sha256=digest(CLUSTER_PROMPT),
            focus='max_unique_support_seed_round_tie', no_pattern='raw_batch_structured_null',
            team_responsibility_selector_used=False)

    def initial_state(self):
        return None

    def prepare(self, context, state, provider):
        full = dict(errors=[raw_error(r, context.observations[r['stable_example_id']]) for r in context.rows],
                    structured_feedback=discover(context, provider))
        selected, focus = project(full, context.seed, context.ordinal)
        return PreparedFeedback(selected['errors'], selected['structured_feedback'], None,
            dict(focus=focus, complete_diagnosis=full), None)

    def observe(self, outcome, state):
        return state


class GradientOnlyModule(LargestPatternModule):
    """Actual formal diagnostics; clustering is explicitly omitted, not failed."""
    def binding(self):
        value=super().binding()
        value.update(identity='ABLATION_FORMAL_GRADIENT_ONLY_V1',
            implementation='ablations.pattern.GradientOnlyModule',semantic_calls_per_opportunity=9,
            focus='all_actionable_gradients_no_cluster',pattern_status='MODULE_DISABLED')
        return value

    def prepare(self,context,state,provider):
        structured=discover(context,provider,clustering=False)
        return PreparedFeedback([raw_error(r,context.observations[r['stable_example_id']])
            for r in context.rows],structured,None,dict(pattern_status='MODULE_DISABLED'),None)
