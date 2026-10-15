"""Three independent single-step repairs from an opportunity-start population."""
from dataclasses import asdict, dataclass
import json
from .. import current_contract as versions
from ..local_optimizers.schemas import LocalEvidenceExample
from .schemas import SearchContractError, SearchCandidate, SearchResult
from .parent_population import digest
from .system_prompt import block_lineage
from .reflection_steps import (FrozenParentSelection,TeamGroupChannel,ReflectionEvidenceComposition,
    SingleStepReflection,SlotGeneration)


@dataclass(frozen=True)
class PopulationSearchConfig:
    identity_version: str = versions.CURRENT_LAYER1_SEARCH_VERSION
    optimizer_input_schema: str = versions.CURRENT_OPTIMIZER_INPUT_SCHEMA
    offspring_slots: int = 3
    maximum_parents: int = 3
    structural_draws_per_slot: int = 2
    maximum_support_examples: int = 3
    candidate_contract: str = versions.SEMANTIC_MUTABLE_CONTRACT_VERSION

    def __post_init__(self):
        if asdict(self) != {k:v.default for k,v in self.__dataclass_fields__.items()}:
            raise SearchContractError('POPULATION_SEARCH_CONFIG_MISMATCH')

    def identity(self): return digest(asdict(self))


INSTRUCTION = '''Perform ONE evidence-grounded edit to the selected Parent Prompt.
Layer2 has specified a repair hypothesis and representative erroneous samples.
The selected Pattern is the primary repair target. Observations carry explicit
applies_to labels for root and/or selected_parent. Root diagnostics concern the
Root Prompt; selected_parent responses belong to the Parent you are editing; do not confuse
them, invent hidden reasoning, or regenerate Responsibility. A Pattern is a
hypothesis, not a universally proven rule. Use the complete readable evidence and
worked reference to formulate a specific conditional repair. Scoped experience
describes previous complete edits in their measured contexts, not causal laws.
Generate a replacement for exactly one role/strategy/answer block. Extract useful
domain information and strategies from the supplied training feedback. There is no
character budget. An unchanged replacement is recorded and skipped without evaluation.
Return exactly {"target_block":"strategy","new_content":"complete replacement block",
"change_summary":"intended change"}. Choose the block from
the evidence. No decision field or other keys. Fields must be strings; target_block
must be exactly role, strategy or answer. Truncation is labeled; do not infer omitted
content. Format-recovery feedback is not mathematical evidence or evaluation.
Group experience is auxiliary observation and tentative guidance from the Root
Team, not causality, a new Pattern, or Parent-specific Responsibility. Positive
experience may suggest useful changes; negative experience may suggest ineffective
or risky directions, not hard prohibitions. Use only relevant insights; you need
not adopt every suggestion. If Parent differs from Root, check its actual responses
for applicability. At inference the Solver receives only its edited Prompt and
the raw problem, with no team/optimizer evidence channel.
'''


class CurrentOptimizer:
    def __init__(self, *, evaluator, reflection_lm, accounting_reader, run_root,
            parent_selection=None,group_channel=None,evidence_composition=None,candidate_generation=None):
        self.parent_selection=parent_selection or FrozenParentSelection()
        self.group_channel=group_channel or TeamGroupChannel()
        self.evidence_composition=evidence_composition or ReflectionEvidenceComposition()
        self.candidate_generation=candidate_generation or SingleStepReflection()
        self.config = PopulationSearchConfig()
        self.evaluator = evaluator; self.reflection_lm = reflection_lm
        self.accounting_reader = accounting_reader; self.run_root = run_root
        self.population = None; self.profile_reader = None; self.memory = None
        self.execution_observer = None
        self.group_evidence = None; self.group_provider = None

    async def search(self, opportunity, context, seed):
        patterns = context.pattern_view
        pattern = next((p for p in patterns.get('patterns', ())
            if p['pattern_id'] == patterns.get('focus_mechanism_id')), None)
        if pattern is None:
            return SearchResult((), 'NO_ACTIONABLE_PATTERN')
        if self.population is None or self.profile_reader is None or self.memory is None:
            raise SearchContractError('POPULATION_PORTS_NOT_BOUND')
        rows = opportunity.evidence.repair_examples
        if not 1 <= len(rows) <= 3 or any(r.source_split != 'optimize'
                or r.example_id not in pattern['support_ids'] for r in rows):
            raise SearchContractError('PATTERN_SUPPORT_EVIDENCE_INVALID')
        parents, plan = self.parent_selection.select(self.population,opportunity.target_member,
            seed=seed, opportunity_id=opportunity.opportunity_id)
        if self.group_evidence is None or self.group_provider is None:
            raise SearchContractError('GROUP_DIAGNOSIS_PORTS_NOT_BOUND')
        canonical_group,group_experience,group_audit,private_root=self.group_channel.deliver(
            evidence_source=self.group_evidence,provider=self.group_provider,opportunity=opportunity,
            pattern=pattern,rows=rows,run_root=self.run_root,observer=self.execution_observer)
        from ..persistence.durable_io import atomic_write_json
        diagnostics = {r['example_id']:r for r in patterns.get('gradient_diagnostics', ())
            if r.get('disposition') == 'ACTIONABLE'}
        seen = {p.identity for p in self.population.ordered(opportunity.target_member)}
        candidates = []; outcomes = []; draws = 0
        before = self.accounting_reader()
        for slot, parent in enumerate(parents):
            packet,view,view_stats,private=self.evidence_composition.compose(
                opportunity=opportunity,parent=parent,rows=rows,pattern=pattern,diagnostics=diagnostics,
                canonical_group=canonical_group,group_experience=group_experience,
                profile_reader=self.profile_reader,memory=self.memory,config=self.config)
            slot_before=self.accounting_reader()
            seen_before=set(seen)
            generated=self.candidate_generation.generate(reflection_lm=self.reflection_lm,config=self.config,
                opportunity=opportunity,parent=parent,rows=rows,pattern=pattern,plan=plan,slot=slot,
                packet=packet,view=view,private=private,canonical_group=canonical_group,seen=seen)
            if not isinstance(generated,SlotGeneration):raise SearchContractError('REFLECTION_COMPONENT_OUTPUT_INVALID')
            candidate=generated.candidate;errors=generated.errors;wire_sizes=generated.wire_sizes
            if candidate is not None:
                from .reflection_steps import candidate_legality
                if candidate_legality(candidate.prompt,parent=parent,rows=rows,private=private,
                        canonical_group=canonical_group,seen=seen_before):
                    raise SearchContractError('REFLECTION_COMPONENT_UNSAFE_OUTPUT')
                if len(block_lineage(parent.prompt,candidate.prompt)['edited_blocks'])!=1:
                    if candidate.prompt==parent.prompt:
                        candidate=None
                        generated=SlotGeneration(None,errors,generated.draws,wire_sizes,'NO_OP')
                    else:raise SearchContractError('REFLECTION_COMPONENT_EDIT_SCOPE_INVALID')
                if candidate is not None:seen.add(candidate.prompt.prompt_hash)
            draws+=generated.draws
            if candidate is not None:candidates.append(candidate)
            outcome = dict(slot=slot, selected_parent_sha256=parent.identity,
                candidate_id=candidate.candidate_id if candidate else None,
                disposition=generated.disposition, errors=errors,
                reflection_input_sha256=digest(packet))
            slot_after=self.accounting_reader()
            outcome['input_statistics']={**view_stats,'wire_characters_per_draw':wire_sizes,
                'accounting':{k:slot_after[k]-slot_before[k] for k in slot_after}}
            atomic_write_json(private_root/f'slot_{slot}.json',dict(canonical=packet,reflector_view=view,outcome=outcome))
            outcomes.append(outcome)
            if self.execution_observer:
                self.execution_observer('POPULATION_SLOT', dict(packet=packet,reflector_view=view,outcome=outcome))
        after = self.accounting_reader()
        return SearchResult(tuple(candidates), 'THREE_SINGLE_STEP_SLOTS_COMPLETE',
            search_state=dict(selection_plan=plan, generation_outcomes=outcomes,
                reflection_draws=draws, independent_slots=3,group_diagnosis_audit=group_audit), proposal_count=3,
            team_candidate_count=len(candidates), search_meta_calls=after['successful_calls']-before['successful_calls']+group_audit['accounting']['successful_calls'],
            search_meta_tokens=(after['input_tokens']+after['output_tokens']-before['input_tokens']-before['output_tokens']
                +group_audit['accounting']['input_tokens']+group_audit['accounting']['output_tokens']))


class CurrentEngine:
    def __init__(self, optimizer, seed):
        self.optimizer=optimizer; self.seed=seed; self.identity=optimizer.config.identity_version
    async def search(self, opportunity, context):
        return await self.optimizer.search(opportunity, context, self.seed)

CurrentLayer1Config = PopulationSearchConfig
