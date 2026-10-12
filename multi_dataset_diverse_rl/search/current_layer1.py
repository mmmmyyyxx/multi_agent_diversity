"""Three independent single-step repairs from an opportunity-start population."""
from dataclasses import asdict, dataclass
import json
from .. import current_contract as versions
from ..local_optimizers.schemas import LocalEvidenceExample
from .schemas import SearchContractError, SearchCandidate, SearchResult
from .parent_population import digest
from .system_prompt import candidate_failed_checks, block_lineage
from .optimization_evidence import executability_checks
from .optimizer_output_contract import (JSON_INSTRUCTION, parse_output,
    validate_reflection_output, feedback_for_error, reflection_feedback_prefix,
    reject, issue)
from .generation_failures import recoverable_output


@dataclass(frozen=True)
class PopulationSearchConfig:
    identity_version: str = versions.CURRENT_LAYER1_SEARCH_VERSION
    optimizer_input_schema: str = versions.CURRENT_OPTIMIZER_INPUT_SCHEMA
    offspring_slots: int = 3
    maximum_parents: int = 3
    structural_draws_per_slot: int = 2
    maximum_support_examples: int = 3
    max_prompt_chars: int = 3000
    candidate_contract: str = versions.SEMANTIC_MUTABLE_CONTRACT_VERSION

    def __post_init__(self):
        if asdict(self) != {k:v.default for k,v in self.__dataclass_fields__.items()}:
            raise SearchContractError('POPULATION_SEARCH_CONFIG_MISMATCH')

    def identity(self): return digest(asdict(self))


INSTRUCTION = '''Perform ONE evidence-grounded edit to the selected Parent Prompt.
Layer2 has specified a repair hypothesis and representative erroneous samples.
Root diagnostics concern the explicitly labeled root Prompt. Selected-parent
observations are the actual outputs of the Parent you are editing; do not confuse
them, invent hidden reasoning, or regenerate Responsibility. A Pattern is a
hypothesis, not a universally proven rule. Use the complete readable evidence and
worked reference to formulate a specific conditional repair. Scoped experience
describes previous complete edits in their measured contexts, not causal laws.
Generate a concrete edit; abstention is not an output option. Do not copy a problem,
reference, answer or supplied solution into the Prompt. Do not introduce references
to optimizer-only fields. All role/strategy/answer blocks are editable. Change
exactly one block; the entire rendered system Prompt must remain <=3000 characters.
Prefer concise coherent rules; appending redundant checklists is not preservation.
Return exactly {"target_block":"strategy","new_content":"complete replacement block",
"change_summary":"intended change, at most 240 characters"}. Choose the block from
the evidence. No decision field or other keys. Fields must be strings; target_block
must be exactly role, strategy or answer. Truncation is labeled; do not infer omitted
content. Format-recovery feedback is not mathematical evidence or evaluation.
'''


class CurrentOptimizer:
    def __init__(self, *, evaluator, reflection_lm, accounting_reader, run_root):
        self.config = PopulationSearchConfig()
        self.evaluator = evaluator; self.reflection_lm = reflection_lm
        self.accounting_reader = accounting_reader; self.run_root = run_root
        self.population = None; self.profile_reader = None; self.memory = None
        self.execution_observer = None

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
        parents, plan = self.population.select_three(opportunity.target_member,
            seed=seed, opportunity_id=opportunity.opportunity_id)
        self.population.save()
        diagnostics = {r['example_id']:r for r in patterns.get('gradient_diagnostics', ())
            if r.get('disposition') == 'ACTIONABLE'}
        seen = {p.identity for p in self.population.ordered(opportunity.target_member)}
        candidates = []; outcomes = []; draws = 0
        before = self.accounting_reader()
        for slot, parent in enumerate(parents):
            observations = self.profile_reader(opportunity.target_member, parent.prompt, rows)
            examples = []; private = []
            for row in rows:
                s = row.signals; solution = s.get('reference_solution')
                diagnostic = diagnostics.get(row.example_id)
                if not isinstance(solution, str) or not solution.strip() or diagnostic is None:
                    raise SearchContractError('SINGLE_STEP_REPAIR_EVIDENCE_REQUIRED')
                actual = observations[row.example_id]
                examples.append(dict(example_id=row.example_id, problem=s['input_payload'],
                    reference_final_answer=s['gold'],
                    worked_reference=dict(source='dataset_worked_solution', split='optimize',
                        text=solution[:4096], original_characters=len(solution), truncated=len(solution)>4096),
                    root_observation=dict(prompt_sha256=opportunity.parent_prompt.prompt_hash,
                        trajectory=s['solver_trajectory'], correct=s['target_member_correct']),
                    accepted_root_diagnostic={k:diagnostic[k] for k in
                        ('observed_failure','diagnosis','reusable_correction','suggested_block','expected_effect')},
                    selected_parent_observation=actual))
                private.extend((solution, s['solver_trajectory'].get('observed_response', ''),
                    actual['trajectory'].get('observed_response', ''), diagnostic['observed_failure'], diagnostic['diagnosis']))
            memory = self.memory.read_for_member(opportunity.target_member, parent.identity)
            packet = dict(schema=self.config.optimizer_input_schema,
                selected_parent=parent.prompt.to_dict(), selected_parent_sha256=parent.identity,
                repair_direction=dict(pattern_id=pattern['pattern_id'],
                    generalized_gradient=pattern['generalized_gradient'],
                    evidence_root_prompt_sha256=opportunity.parent_prompt.prompt_hash),
                repair_examples=examples, scoped_experience=memory,
                edit_capacity=dict(maximum_rendered_characters=3000,
                    maximum_replacement_block_characters={k:3000-4-sum(len(getattr(parent.prompt,j))
                        for j in ('role','strategy','answer') if j!=k) for k in ('role','strategy','answer')}))
            feedback = None; errors = []; candidate = None
            for draw in range(self.config.structural_draws_per_slot):
                draws += 1
                try:
                    text = self.reflection_lm(reflection_feedback_prefix(feedback) + INSTRUCTION
                        + JSON_INSTRUCTION + json.dumps(packet, sort_keys=True, separators=(',', ':'), ensure_ascii=True))
                    value = validate_reflection_output(parse_output(text, stage='reflection'))
                    if value['new_content'].strip()==getattr(parent.prompt,value['target_block']).strip():
                        reject('REFLECTION_OUTPUT_CONTRACT_INVALID','reflection',[issue('new_content','CONTENT_REJECTED')])
                    try: child = parent.prompt.edit(value['target_block'], value['new_content'])
                    except SearchContractError as error:
                        reject('REFLECTION_OUTPUT_CONTRACT_INVALID', 'reflection',
                            [issue('new_content', 'OVER_LIMIT' if str(error)=='SYSTEM_PROMPT_OVER_LENGTH' else 'CONTENT_REJECTED')])
                    check_examples = tuple(LocalEvidenceExample(row.example_id,
                        row.signals['input_payload'], row.signals['gold']) for row in rows)
                    failed = candidate_failed_checks(child, parent_prompt=parent.prompt, examples=check_examples)
                    failed += executability_checks(child.render(), private_texts=private)
                    if failed or child.prompt_hash in seen:
                        reject('REFLECTION_OUTPUT_CONTRACT_INVALID', 'reflection', [issue('new_content', 'CONTENT_REJECTED')])
                    seen.add(child.prompt_hash)
                    lineage = block_lineage(parent.prompt, child)
                    candidate = SearchCandidate(f'population:{slot}:{child.prompt_hash[:12]}', child,
                        lineage=lineage, backend_details=dict(changed=True, contract_valid=True,
                            opportunity_id=opportunity.opportunity_id, parent_state_id=opportunity.parent_state_id,
                            slot=slot, selected_parent_sha256=parent.identity,
                            selected_parent=parent.prompt.to_dict(), population_revision=plan['population_revision'],
                            pattern_id=pattern['pattern_id'],repair_hypothesis=pattern['generalized_gradient'], support_ids=[r.example_id for r in rows],
                            intended_change=value['change_summary'], reflection_input_sha256=digest(packet)))
                    candidates.append(candidate); break
                except SearchContractError as error:
                    if not recoverable_output(error, 'reflection'): raise
                    feedback = feedback_for_error(error, stage='reflection')
                    errors.append(dict(draw=draw, category=str(error), feedback=feedback))
            outcome = dict(slot=slot, selected_parent_sha256=parent.identity,
                candidate_id=candidate.candidate_id if candidate else None,
                disposition='GENERATED' if candidate else 'GENERATION_FAILED', errors=errors,
                reflection_input_sha256=digest(packet))
            outcomes.append(outcome)
            if self.execution_observer:
                self.execution_observer('POPULATION_SLOT', dict(packet=packet, outcome=outcome))
        after = self.accounting_reader()
        return SearchResult(tuple(candidates), 'THREE_SINGLE_STEP_SLOTS_COMPLETE',
            search_state=dict(selection_plan=plan, generation_outcomes=outcomes,
                reflection_draws=draws, independent_slots=3), proposal_count=3,
            team_candidate_count=len(candidates), search_meta_calls=after['successful_calls']-before['successful_calls'],
            search_meta_tokens=(after['input_tokens']+after['output_tokens']-before['input_tokens']-before['output_tokens']))


class CurrentEngine:
    def __init__(self, optimizer, seed):
        self.optimizer=optimizer; self.seed=seed; self.identity=optimizer.config.identity_version
    async def search(self, opportunity, context):
        return await self.optimizer.search(opportunity, context, self.seed)

CurrentLayer1Config = PopulationSearchConfig
