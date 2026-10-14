"""Measurement provenance, pool updates and final admission are distinct decisions."""
from .parent_population import Fitness

class CompleteOptimizeFitness:
    identity='COMPLETE_OPTIMIZE_MEASURED_FITNESS_V1'
    def assign(self,state,member,source,observation_id,count):
        correct=state.team_scores['vote_correct_count'] if source=='TEAM_VOTE' else state.member_scores[member]
        return Fitness(source,int(correct),count,state.diagnostics['evaluation_support_identity'],
            state.team_state_id,tuple(p.prompt_hash for p in state.member_prompts),observation_id)

class MeasuredPoolUpdate:
    identity='MEASURED_CHILD_POOL_UPSERT_V1'
    def update(self,pool,member,prompt,vote,proxy):return pool.update(member,prompt,vote,proxy)

class JointFinalAdmission:
    identity='JOINT_INITIAL_FLOOR_VOTE_NONREGRESSION_V1'
    def allowed(self,*,child,baseline,changed,initial_scores,invalid,parent_invalid,invalidity):
        return (all(child.member_scores[m]>=initial_scores[m] for m in changed)
            and child.team_scores['vote_correct_count']>=baseline.team_scores['vote_correct_count']
            and (invalidity or invalid<=parent_invalid))
