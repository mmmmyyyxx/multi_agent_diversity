"""Single-step team repair components; no evaluation or deployment authority."""
from dataclasses import dataclass
import json
from ..local_optimizers.schemas import LocalEvidenceExample
from .schemas import SearchContractError, SearchCandidate
from .parent_population import digest
from .system_prompt import candidate_failed_checks, block_lineage, replacement_budgets
from .optimization_evidence import executability_checks
from .optimizer_output_contract import (JSON_INSTRUCTION,parse_output,validate_reflection_output,
    feedback_for_error,reflection_feedback_prefix,reject,issue)
from .generation_failures import recoverable_output
from .group_diagnosis import GroupDiagnoser,KINDS
from .reflector_view import project_reflector

class FrozenParentSelection:
    identity='OPPORTUNITY_START_THREE_FROZEN_PARENTS_V1'
    def select(self,population,member,*,seed,opportunity_id):
        parents,plan=population.select_three(member,seed=seed,opportunity_id=opportunity_id)
        population.save()
        return parents,plan

class TeamGroupChannel:
    identity='ROOT_TEAM_GROUP_CHANNEL_V1'
    def generate(self,provider,canonical_group,rows,private_root):
        from ..persistence.durable_io import atomic_write_json
        return GroupDiagnoser(provider,audit_writer=lambda record:
            atomic_write_json(private_root/f"group_draw_{record['attempt']['draw']}.json",record)).diagnose(canonical_group,rows)
    def deliver(self,*,evidence_source,provider,opportunity,pattern,rows,run_root,observer):
        canonical_group=evidence_source.freeze(opportunity,pattern)
        from ..persistence.durable_io import atomic_write_json
        from pathlib import Path
        private_root=Path(run_root)/'group_reflection.private'/digest(opportunity.opportunity_id)
        atomic_write_json(private_root/'root_evidence.json',canonical_group)
        group_before=provider.accounting()
        try:
            group_experience,group_audit=self.generate(provider,canonical_group,rows,private_root)
        except BaseException as error:
            atomic_write_json(private_root/'group_failure.json',dict(category=type(error).__name__,reason=str(error),
                resource_statistics=getattr(error,'resource_statistics',None),accounting=provider.accounting()))
            raise
        group_after=provider.accounting()
        group_audit['accounting']={k:group_after[k]-group_before[k] for k in group_after}
        atomic_write_json(private_root/'group.json',dict(canonical=canonical_group,experience=group_experience,audit=group_audit))
        if observer:
            observer('GROUP_DIAGNOSIS',dict(canonical=canonical_group,experience=group_experience,audit=group_audit))
        return canonical_group,group_experience,group_audit,private_root

class DisabledGroupChannel(TeamGroupChannel):
    """No diagnosis is claimed; retain existing root evidence and safety boundaries."""
    identity='EXPLICITLY_DISABLED_GROUP_CHANNEL_V1'
    def generate(self,provider,canonical_group,rows,private_root):
        return dict(status='MODULE_DISABLED',module_identity=self.identity,**{k:None for k in KINDS}),dict(
            status='MODULE_DISABLED',module_identity=self.identity,input_support_count=len(rows),
            nonempty_experiences=0,structural_draws=0,structural_errors=[])


def worked_reference(solution):
    """Actual formal worked-solution view; original source remains private."""
    if not isinstance(solution,str) or not solution.strip():
        raise SearchContractError('SINGLE_STEP_REPAIR_EVIDENCE_REQUIRED')
    return dict(source='dataset_worked_solution',split='optimize',text=solution[:4096],
        original_characters=len(solution),truncated=len(solution)>4096)

class ReflectionEvidenceComposition:
    identity='READABLE_SINGLE_STEP_EVIDENCE_COMPOSITION_V1'
    def compose(self,*,opportunity,parent,rows,pattern,diagnostics,canonical_group,
                group_experience,profile_reader,memory,config):
        observations = profile_reader(opportunity.target_member, parent.prompt, rows)
        examples = []; private = []
        for row in rows:
            s = row.signals; solution = s.get('reference_solution')
            diagnostic = diagnostics.get(row.example_id)
            if not isinstance(solution, str) or not solution.strip() or diagnostic is None:
                raise SearchContractError('SINGLE_STEP_REPAIR_EVIDENCE_REQUIRED')
            actual = observations[row.example_id]
            examples.append(dict(example_id=row.example_id, problem=s['input_payload'],
                reference_final_answer=s['gold'],
                worked_reference=worked_reference(solution),
                root_observation=dict(prompt_sha256=opportunity.parent_prompt.prompt_hash,
                    trajectory=s['solver_trajectory'], correct=s['target_member_correct']),
                accepted_root_diagnostic={k:diagnostic[k] for k in
                    ('observed_failure','diagnosis','reusable_correction','suggested_block','expected_effect')},
                selected_parent_observation=actual))
            private.extend((solution, s['solver_trajectory'].get('observed_response', ''),
                actual['trajectory'].get('observed_response', ''), diagnostic['observed_failure'], diagnostic['diagnosis']))
        memory = memory.read_for_member(opportunity.target_member, parent.identity)
        packet = dict(schema=config.optimizer_input_schema,
            selected_parent=parent.prompt.to_dict(), selected_parent_sha256=parent.identity,
            repair_direction=dict(pattern_id=pattern['pattern_id'],
                generalized_gradient=pattern['generalized_gradient'],
                evidence_root_prompt_sha256=opportunity.parent_prompt.prompt_hash),
            repair_examples=examples, scoped_experience=memory,
            group_experience=group_experience,
            edit_capacity=dict(maximum_rendered_characters=3000,
                maximum_replacement_block_characters=replacement_budgets(parent.prompt)))
        view,view_stats=project_reflector(packet,root_prompt=opportunity.parent_prompt,
            selected_parent=parent.prompt,target_member=opportunity.target_member)
        private.extend(p['trajectory']['observed_response'] for e in canonical_group['examples'] for p in e['members'])
        return packet,view,view_stats,private

@dataclass(frozen=True)
class SlotGeneration:
    candidate: object
    errors: list
    draws: int
    wire_sizes: list
    disposition: str

def inference_dependency_checks(text):
    """Shared formal guard: inference cannot access optimizer-only team evidence."""
    import re
    if re.search(r'\b(?:peer_prompt|peer_answer|root_team|team_snapshot|correct_members|wrong_members)\b|'
            r'\b(?:read|consult|copy|access|use)\b[^.;\n]{0,65}\b(?:peer (?:answers?|prompts?|responses?)|'
            r"other members['’]? (?:answers?|prompts?|responses?)|group diagnosis|team snapshot)\b",
            text,re.I):return ('inference_unavailable_group_dependency',)
    return ()


def candidate_legality(child,*,parent,rows,private,canonical_group,seen):
    """Shared output boundary; replacing a generator does not replace safety."""
    check_examples=tuple(LocalEvidenceExample(row.example_id,row.signals['input_payload'],
        row.signals['gold']) for row in rows)
    failed=candidate_failed_checks(child,parent_prompt=parent.prompt,examples=check_examples)
    if failed:return failed
    failed+=executability_checks(child.render(),private_texts=private)
    failed+=inference_dependency_checks(child.render())
    if child.to_dict() in canonical_group['root_prompts'] and child!=parent.prompt:
        failed+=('peer_prompt_copy',)
    if child.prompt_hash in seen:failed+=('duplicate_prompt',)
    return failed

class SingleStepReflection:
    identity='BOUNDED_SINGLE_STEP_REFLECTION_V1'
    def generate(self,*,reflection_lm,config,opportunity,parent,rows,pattern,plan,slot,
                 packet,view,private,canonical_group,seen):
        from .current_layer1 import INSTRUCTION
        feedback = None; errors = []; candidate = None; draws=0;wire_sizes=[]
        for draw in range(config.structural_draws_per_slot):
            draws += 1
            try:
                wire=reflection_feedback_prefix(feedback)+INSTRUCTION+JSON_INSTRUCTION+json.dumps(view,sort_keys=True,separators=(',',':'),ensure_ascii=False)
                wire_sizes.append(len(wire))
                text = reflection_lm(wire)
                value = validate_reflection_output(parse_output(text, stage='reflection'))
                if value['new_content'].strip()==getattr(parent.prompt,value['target_block']).strip():
                    reject('REFLECTION_OUTPUT_CONTRACT_INVALID','reflection',[issue('new_content','CONTENT_REJECTED')])
                try: child = parent.prompt.edit(value['target_block'], value['new_content'])
                except SearchContractError as error:
                    reject('REFLECTION_OUTPUT_CONTRACT_INVALID', 'reflection',
                        [issue('new_content', 'OVER_LIMIT' if str(error)=='SYSTEM_PROMPT_OVER_LENGTH' else 'CONTENT_REJECTED')])
                failed=candidate_legality(child,parent=parent,rows=rows,private=private,
                    canonical_group=canonical_group,seen=seen)
                if failed:
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
                break
            except SearchContractError as error:
                if not recoverable_output(error, 'reflection'): raise
                feedback = feedback_for_error(error, stage='reflection')
                errors.append(dict(draw=draw, category=str(error), feedback=feedback))
        return SlotGeneration(candidate,errors,draws,wire_sizes,'GENERATED' if candidate else 'GENERATION_FAILED')

class DisabledReflection:
    identity='EXPLICITLY_DISABLED_REFLECTION_V1'
    def generate(self,**kwargs):
        return SlotGeneration(None,[],0,[],'MODULE_DISABLED')
