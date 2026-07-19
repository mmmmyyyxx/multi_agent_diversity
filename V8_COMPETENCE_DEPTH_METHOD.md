# V8 Competence-Depth Method

## Coverage Depth

For each question, `Ck` is the fraction of examples with at least `k` correct agents. C1 measures oracle coverage; C2 measures redundant correct coverage. Bottom-2 and mean individual accuracy prevent diversity from being created by sacrificing weak agents.

## Fixed Optimization Probe

V8 samples one competence probe from the optimization training split. Its indices and question hashes are checkpointed. The same probe drives the progressive competence schedule, beam-prompt behavioral profiles, and joint active-team selection. Probe drift is a runtime error.

## Candidate Guards

Candidate quality can use target accuracy, C1/C2 transitions, actual plurality changes, boundary rescue, and shared-error residuals. Invalid output, C1 regression, and catastrophic target-accuracy loss remain hard guards.

## Quality Before Diversity

Joint teams must stay within integer loss tolerances for incumbent plurality, total agent correctness, bottom-2, C1, C2, and every agent. Hierarchical vote, total-correct, bottom-2, C1, and C2 bands then shrink the feasible set. Behavioral complementarity is considered only inside the final band.

## Search-Space Preservation

Safe candidates form a six-item long-term niche archive; three representatives per agent are used for joint enumeration. Novel small-regression branches enter a bounded Probation archive and may reproduce, but cannot be active. Failed initial batches trigger bounded TCS refill with structured rejection feedback. Two deterministic probe folds and two stable snapshots are required before lineage commitment.

## Validation

Validation remains vote and competence first. Stable specialization is only a final tie-break and never overrides plurality or competence quality.
