"""Population strategies operate on measured Parents, never deploy a team."""
import math,random
from .parent_population import digest

class FitnessWeightedSampling:
    identity='EPSILON_SOFTMAX_FITNESS_SAMPLING_V1'
    def probabilities(self,parents):
        best=max(p.fitness.value for p in parents)
        weights=[math.exp(4.0*(p.fitness.value-best)) for p in parents]
        total=math.fsum(weights)
        return tuple(.1/len(parents)+(1-.1)*w/total for w in weights)
    def draw(self,parents,probabilities,*,scope,seed,opportunity_id,member):
        rng=random.Random(int(digest([scope,seed,opportunity_id,member]),16));chosen=[]
        for _ in range(3):
            x=rng.random();cumulative=0.;selected=parents[-1]
            for parent,probability in zip(parents,probabilities,strict=True):
                cumulative+=probability
                if x<cumulative:selected=parent;break
            chosen.append(selected)
        return tuple(chosen)

class BestThreeRetention:
    identity='TOP_THREE_FITNESS_HASH_RETENTION_V1'
    def keep(self,ordered):return ordered[:3]

class PerMemberArgmaxAssembly:
    identity='PER_MEMBER_ARGMAX_ASSEMBLY_SELECTION_V1'
    def select(self,ordered_members):return tuple(rows[0] for rows in ordered_members)
