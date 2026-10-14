"""Bounded, provenance-bearing per-member populations, never a deployed team.

Selection plans freeze before any offspring is generated. Checkpoints are private
and identity-bound; loading one does not authorize paid execution or replay.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import re

from .schemas import SearchContractError
from .system_prompt import SystemPrompt, require_prompt
from ..persistence.durable_io import atomic_write_json, read_json

POLICY = dict(identity='VOTE_GUIDED_PARENT_POPULATION_V1', members=5,
    maximum_parents_per_member=3, offspring_slots=3, sampling='with_replacement',
    selection='epsilon_mixed_stable_softmax', inverse_temperature=4.0,
    uniform_mixture=0.1, tie='canonical_prompt_sha256_ascending',
    selection_snapshot='opportunity_start_no_new_child_parents',
    fitness='last_actual_optimize_vote_else_complete_member_accuracy',
    normalization='correct_count_over_frozen_complete_optimize_count',
    score_context='explicit_team_state_and_ordered_prompt_hashes',
    eviction='lowest_current_fitness_no_protected_parent',
    final_selection='per_member_argmax_at_search_closure',
    final_evaluation='joint_complete_optimize_and_private_shadow_no_silent_fallback')
SCHEMA = 'VOTE_PARENT_POPULATION_CHECKPOINT_V1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Fitness:
    source: str
    correct: int
    count: int
    membership_sha256: str
    team_state_id: str
    team_prompt_hashes: tuple[str, ...]
    observation_id: str
    split: str = 'optimize'

    def __post_init__(self):
        if (self.source not in {'TEAM_VOTE', 'MEMBER_ACCURACY_PROXY'}
                or type(self.correct) is not int or type(self.count) is not int
                or not 0 <= self.correct <= self.count or self.count <= 0
                or self.split != 'optimize' or not self.observation_id
                or not self.team_state_id or len(self.team_prompt_hashes) != 5
                or any(not isinstance(h, str) or re.fullmatch('[a-f0-9]{64}',h) is None for h in self.team_prompt_hashes)
                or not isinstance(self.membership_sha256, str) or re.fullmatch('[a-f0-9]{64}',self.membership_sha256) is None):
            raise SearchContractError('POPULATION_FITNESS_PROVENANCE_INVALID')

    @property
    def value(self):
        return self.correct / self.count

    @classmethod
    def load(cls, value):
        try:
            return cls(**{**value, 'team_prompt_hashes':tuple(value['team_prompt_hashes'])})
        except (KeyError, TypeError, ValueError) as error:
            raise SearchContractError('POPULATION_FITNESS_PROVENANCE_INVALID') from error


@dataclass(frozen=True)
class Parent:
    prompt: SystemPrompt
    fitness: Fitness
    member_accuracy: Fitness

    @property
    def identity(self):
        return self.prompt.prompt_hash


class ParentPopulation:
    def __init__(self, *, scope, checkpoint: Path | None = None,sampling=None,retention=None,assembly=None):
        from .population_policies import FitnessWeightedSampling,BestThreeRetention,PerMemberArgmaxAssembly
        self.sampling=sampling or FitnessWeightedSampling()
        self.retention=retention or BestThreeRetention()
        self.assembly=assembly or PerMemberArgmaxAssembly()
        if (not isinstance(scope, dict) or not scope.get('method_identity')
                or not isinstance(scope.get('support_sha256'),str)
                or re.fullmatch('[a-f0-9]{64}',scope['support_sha256']) is None
                or type(scope.get('support_count')) is not int or scope['support_count']<=0):
            raise SearchContractError('POPULATION_SCOPE_REQUIRED')
        self.scope = deepcopy(scope)
        self.checkpoint = Path(checkpoint) if checkpoint is not None else None
        self.members = {m: {} for m in range(5)}
        self.revision = 0
        self.plans = {}

    def _validate(self, parent, member):
        require_prompt(parent.prompt)
        if (parent.fitness.membership_sha256 != self.scope['support_sha256']
                or parent.member_accuracy.membership_sha256 != self.scope['support_sha256']
                or parent.fitness.count != self.scope['support_count']
                or parent.member_accuracy.count != self.scope['support_count']
                or parent.member_accuracy.source != 'MEMBER_ACCURACY_PROXY'
                or parent.fitness.team_prompt_hashes[member]!=parent.identity
                or parent.member_accuracy.team_prompt_hashes[member]!=parent.identity):
            raise SearchContractError('POPULATION_FITNESS_SCOPE_MISMATCH')

    def ordered(self, member):
        if type(member) is not int or member not in self.members:
            raise SearchContractError('POPULATION_MEMBER_INVALID')
        return tuple(sorted(self.members[member].values(),
            key=lambda p: (-p.fitness.value, p.identity)))

    def update(self, member, prompt, fitness, member_accuracy):
        self.ordered(member)
        parent = Parent(require_prompt(prompt), fitness, member_accuracy)
        self._validate(parent,member)
        existing = self.members[member].get(parent.identity)
        # Re-reading an individual score must not replace a measured Vote.
        if existing and existing.fitness.source == 'TEAM_VOTE' and fitness.source != 'TEAM_VOTE':
            parent = Parent(parent.prompt, existing.fitness, member_accuracy)
        staged={**self.members[member],parent.identity:parent}
        ordered=tuple(sorted(staged.values(),key=lambda p:(-p.fitness.value,p.identity)))
        kept=tuple(self.retention.keep(ordered))
        if (not 1<=len(kept)<=3 or len({p.identity for p in kept})!=len(kept)
                or any(p not in ordered for p in kept)):
            raise SearchContractError('POPULATION_RETENTION_OUTPUT_INVALID')
        evicted = sorted(set(staged) - {p.identity for p in kept})
        self.members[member] = {p.identity:p for p in kept}
        self.revision += 1
        return dict(retained=parent.identity in self.members[member], evicted=evicted,
            source=parent.fitness.source, fitness=parent.fitness.value, revision=self.revision)

    def probabilities(self, member):
        parents = self.ordered(member)
        if not parents:
            raise SearchContractError('POPULATION_NOT_BOOTSTRAPPED')
        probabilities=tuple(self.sampling.probabilities(parents))
        if (len(probabilities)!=len(parents) or any(type(q) not in (int,float)
                or not math.isfinite(q) or not 0<q<=1 for q in probabilities)
                or abs(math.fsum(probabilities)-1)>1e-12):
            raise SearchContractError('POPULATION_SAMPLING_OUTPUT_INVALID')
        return parents,probabilities

    def select_three(self, member, *, seed, opportunity_id):
        if type(seed) is not int or not opportunity_id:
            raise SearchContractError('POPULATION_SELECTION_IDENTITY_INVALID')
        key = digest([member, seed, opportunity_id])
        if key in self.plans:
            raise SearchContractError('POPULATION_OPPORTUNITY_ALREADY_SELECTED')
        parents, probabilities = self.probabilities(member)
        chosen=self.sampling.draw(parents,probabilities,scope=self.scope,seed=seed,
            opportunity_id=opportunity_id,member=member)
        chosen=tuple(chosen)
        if len(chosen)!=3 or any(p not in parents for p in chosen):
            raise SearchContractError('POPULATION_DRAW_OUTPUT_INVALID')
        plan = dict(member=member, seed=seed, opportunity_id=opportunity_id,
            population_revision=self.revision,
            distribution=[dict(prompt_sha256=p.identity, probability=q, fitness=asdict(p.fitness))
                for p,q in zip(parents, probabilities, strict=True)],
            selected_prompt_sha256=[p.identity for p in chosen])
        self.plans[key] = plan
        return tuple(chosen), deepcopy(plan)

    def best_team(self):
        if any(not self.members[m] for m in range(5)):
            raise SearchContractError('POPULATION_NOT_BOOTSTRAPPED')
        parents = self.assembly.select(tuple(self.ordered(m) for m in range(5)))
        parents=tuple(parents)
        if len(parents)!=5 or any(p not in self.ordered(m) for m,p in enumerate(parents)):
            raise SearchContractError('POPULATION_ASSEMBLY_OUTPUT_INVALID')
        return tuple(p.prompt for p in parents), dict(population_revision=self.revision,
            selected=[dict(member=m, prompt_sha256=p.identity, fitness=asdict(p.fitness))
                for m,p in enumerate(parents)], selection='PER_MEMBER_ARGMAX_BEFORE_JOINT_EVALUATION')

    def export(self):
        payload = dict(schema=SCHEMA, scope=self.scope, revision=self.revision,
            members={str(m):[dict(prompt=p.prompt.to_dict(), prompt_sha256=p.identity,
                fitness=asdict(p.fitness), member_accuracy=asdict(p.member_accuracy))
                for p in self.ordered(m)] for m in range(5)}, plans=self.plans)
        return json.loads(json.dumps({**payload, 'integrity_sha256':digest(payload)},allow_nan=False))

    def save(self):
        if self.checkpoint is not None:
            atomic_write_json(self.checkpoint, self.export())

    def restore(self, payload):
        try:
            checksum = payload['integrity_sha256']
            body = {k:v for k,v in payload.items() if k != 'integrity_sha256'}
            if set(body)!={'schema','scope','revision','members','plans'}:
                raise SearchContractError('POPULATION_CHECKPOINT_INVALID')
            if digest(body) != checksum or body['schema'] != SCHEMA or body['scope'] != self.scope:
                raise SearchContractError('POPULATION_CHECKPOINT_IDENTITY_MISMATCH')
            if set(body['members']) != set(map(str, range(5))) or type(body['revision']) is not int or body['revision'] < 0:
                raise SearchContractError('POPULATION_CHECKPOINT_INVALID')
            members = {}
            for member in range(5):
                rows = body['members'][str(member)]
                if not 1 <= len(rows) <= 3:
                    raise SearchContractError('POPULATION_CHECKPOINT_INVALID')
                members[member] = {}
                for row in rows:
                    if set(row)!={'prompt','prompt_sha256','fitness','member_accuracy'}:
                        raise SearchContractError('POPULATION_CHECKPOINT_INVALID')
                    prompt = SystemPrompt.from_dict(row['prompt'])
                    parent = Parent(prompt, Fitness.load(row['fitness']), Fitness.load(row['member_accuracy']))
                    self._validate(parent,member)
                    if prompt.prompt_hash != row['prompt_sha256'] or parent.identity in members[member]:
                        raise SearchContractError('POPULATION_CHECKPOINT_INVALID')
                    members[member][parent.identity] = parent
            if not isinstance(body['plans'], dict):
                raise SearchContractError('POPULATION_CHECKPOINT_INVALID')
            for key,plan in body['plans'].items():
                if (set(plan)!={'member','seed','opportunity_id','population_revision','distribution','selected_prompt_sha256'}
                        or type(plan['member']) is not int or plan['member'] not in range(5)
                        or type(plan['seed']) is not int or not plan['opportunity_id']
                        or key!=digest([plan['member'],plan['seed'],plan['opportunity_id']])
                        or type(plan['population_revision']) is not int or not 0<=plan['population_revision']<=body['revision']
                        or not 1<=len(plan['distribution'])<=3
                        or len(plan['selected_prompt_sha256'])!=3):
                    raise SearchContractError('POPULATION_SELECTION_PLAN_INVALID')
                dist=plan['distribution'];ids=[r['prompt_sha256'] for r in dist]
                if any(set(row)!={'prompt_sha256','probability','fitness'} for row in dist):
                    raise SearchContractError('POPULATION_SELECTION_PLAN_INVALID')
                if len(set(ids))!=len(ids) or not set(plan['selected_prompt_sha256'])<=set(ids):
                    raise SearchContractError('POPULATION_SELECTION_PLAN_INVALID')
                fitness=[Fitness.load(r['fitness']) for r in dist]
                if any(f.count!=self.scope['support_count'] or f.membership_sha256!=self.scope['support_sha256']
                        or f.team_prompt_hashes[plan['member']]!=h for f,h in zip(fitness,ids,strict=True)):
                    raise SearchContractError('POPULATION_SELECTION_PLAN_INVALID')
                # Replay the same selected strategy, not a second hardcoded formula.
                from types import SimpleNamespace
                candidates=tuple(SimpleNamespace(fitness=f,identity=h) for f,h in zip(fitness,ids,strict=True))
                probs=self.sampling.probabilities(candidates)
                if any(not isinstance(r['probability'],(float,int)) or not math.isfinite(r['probability'])
                        or abs(r['probability']-p)>1e-15 for r,p in zip(dist,probs,strict=True)):
                    raise SearchContractError('POPULATION_SELECTION_PLAN_INVALID')
                expected=[p.identity for p in self.sampling.draw(candidates,probs,scope=self.scope,
                    seed=plan['seed'],opportunity_id=plan['opportunity_id'],member=plan['member'])]
                if expected!=plan['selected_prompt_sha256']:raise SearchContractError('POPULATION_SELECTION_PLAN_INVALID')
        except (KeyError, TypeError, ValueError) as error:
            raise SearchContractError('POPULATION_CHECKPOINT_INVALID') from error
        # The live object changes only after the entire checkpoint validates.
        self.members = members; self.revision = body['revision']; self.plans = deepcopy(body['plans'])

    def detached(self):
        return ParentPopulation(scope=self.scope,sampling=self.sampling,retention=self.retention,assembly=self.assembly)

    def load(self):
        if self.checkpoint is None:
            raise SearchContractError('POPULATION_CHECKPOINT_PATH_REQUIRED')
        self.restore(read_json(self.checkpoint))
