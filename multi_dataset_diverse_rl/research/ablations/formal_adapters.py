"""Transport/evidence adapters calling production algorithms without copies.

The current team controller, WHO weights and deployment gates are not fabricated.
Independent evidence has an explicit binding; it is not a current team binding.
"""
from copy import deepcopy
import json

from ...benchmarks.math_response_evidence import adaptive_trajectory, trajectory_policy
from ...search.optimization_evidence import POLICY as EVIDENCE_POLICY, GRADIENT_INPUT
from ...search.optimizer_output_contract import parse_output
from ...search.generation_failures import generation_failure
from ...search.schemas import EvidenceItem
from ...search.textual_gradients import (PerExampleGradientProvider, GradientClusterProvider,
    GradientExtractor, REFERENCE_GRADIENT_PROMPT, CLUSTER_PROMPT, normalize_gradient_partition)
from ...search.population_memory import PopulationEditMemory, POLICY as MEMORY_POLICY, LIMITS as MEMORY_LIMITS
from ...current_contract import GRADIENT_PARTITION_COMPLETION_VERSION


class GradientTransportAdapter(PerExampleGradientProvider):
    """Bind independent transport; extract/validate/recover are inherited unchanged."""
    def __init__(self, provider):
        self.broker = provider
        self.prompt = REFERENCE_GRADIENT_PROMPT
        self.calls = 0
        self.input_audit = []
        self.solver_trajectory_policy = trajectory_policy()
        self.optimization_evidence_policy = deepcopy(EVIDENCE_POLICY)
        self.input_schema = GRADIENT_INPUT
        self.numeric_guard_writer = None
        self.recovery_policy = None

    def independent_provider(self):
        return GradientTransportAdapter(self.broker)

    def _call(self, payload):
        self.calls += 1
        self.last_response = self.broker.complete(role=self.role, split='optimize', stage=self.stage,
            messages=[dict(role='system', content=self.prompt),
                      dict(role='user', content=json.dumps(payload, sort_keys=True, ensure_ascii=True))])
        return parse_output(self.last_response['text'], stage='gradient')


class ClusterTransportAdapter(GradientClusterProvider):
    """cluster(), partition validation and bounded retries are production methods."""
    def __init__(self, provider):
        super().__init__(provider, prompt=CLUSTER_PROMPT,
            partition_completion_policy=GRADIENT_PARTITION_COMPLETION_VERSION)

    def _call(self, payload):
        self.calls += 1
        self.last_response = self.broker.complete(role=self.role, split='optimize', stage=self.stage,
            messages=[dict(role='system', content=self.prompt),
                      dict(role='user', content=json.dumps(payload, sort_keys=True, ensure_ascii=True))])
        try:
            return json.loads(self.last_response['text'])
        except (ValueError, TypeError) as exc:
            raise generation_failure('PATTERN_GRADIENT_CLUSTER_INVALID') from exc


def discover(context, provider, *, clustering=True):
    evidence = []
    for row in context.rows:
        ident = row['stable_example_id']
        observed = context.observations[ident]
        evidence.append(EvidenceItem(ident, 'optimize', frozenset({'pattern'}), dict(
            input_payload=row['content']['problem'], gold=row['reference_final_answer'],
            reference_solution=row['content']['solution'], target_output=observed['answer'],
            target_member_correct=False, target_member_valid=observed['valid'],
            correctness_signal_identity='target_member_correctness_validity_v1', responsibility_labels=(),
            solver_trajectory=adaptive_trajectory(observed['profile'], member_id=0,
                prompt=context.prompt, example_id=ident, problem=row['content']['problem']))))
    extractor = GradientExtractor(GradientTransportAdapter(provider))
    cluster = ClusterTransportAdapter(provider)
    gradients = extractor.extract(context.prompt, tuple(evidence)) if evidence else ()
    partition = dict(patterns=[], unassigned_ids=[])
    if gradients and clustering:
        actionable = {g['example_id'] for g in gradients}
        partition = cluster.cluster(dict(gradients=list(gradients)),
            evidence_rows=tuple(e for e in evidence if e.example_id in actionable))
    # Normalization is authoritative and has no team ranking or fabricated labels.
    normalized=[]
    if gradients and clustering:
        result=normalize_gradient_partition(partition,
            tuple(e for e in evidence if e.example_id in actionable),gradients)
        normalized=[dict(pattern_id=p['pattern_id'],generalized_gradient=p['generalized_gradient'],
            support_ids=list(p['support_ids'])) for p in sorted(result['patterns'],key=lambda p:p['pattern_id'])]
    return dict(gradients=list(gradients), diagnostics=getattr(extractor, 'evidence_diagnostics', []),
        patterns=normalized, unassigned_ids=partition['unassigned_ids'],
        partition_audit=cluster.partition_audit,
        cluster_structural_errors=getattr(cluster, 'structural_errors', []))


def production_memory_binding():
    return dict(implementation='multi_dataset_diverse_rl.search.population_memory.PopulationEditMemory',
        policy=deepcopy(MEMORY_POLICY), limits=deepcopy(MEMORY_LIMITS),
        status='ADAPTATION_POLICY_REQUIRED',
        unresolved=['single_prompt_bootstrap_scope', 'absent_team_vote_and_shadow_fields',
                    'which_candidate_outcomes_to_record', 'optimizer_delivery_contract'])


def new_production_memory():
    """Return the actual implementation; no substitute storage/retrieval algorithm."""
    return PopulationEditMemory()
