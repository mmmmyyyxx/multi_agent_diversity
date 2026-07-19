# V8 Competence-Depth Method

## Coverage Depth

For each question, `Ck` is the fraction of examples with at least `k` correct agents. C1 measures oracle coverage; C2 measures redundant correct coverage. Bottom-2 and mean individual accuracy prevent diversity from being created by sacrificing weak agents.

## Fixed Optimization Probe

V8 samples one competence probe from the optimization training split. Its indices and question hashes are checkpointed. The same probe drives the progressive competence schedule, beam-prompt behavioral profiles, and joint active-team selection. Probe drift is a runtime error.

## Candidate Guards

Candidate quality can use target accuracy, C1/C2 transitions, actual plurality changes, boundary rescue, and shared-error residuals. Invalid output, C1 regression, and catastrophic target-accuracy loss remain hard guards.

## Quality Before Diversity

Joint teams must stay within one-question tolerances of incumbent plurality, mean, bottom-2, C1, and C2, plus a per-agent accuracy tolerance. An epsilon-Pareto frontier is formed from feasible teams. Behavioral complementarity is considered only inside that frontier.

## Validation

Validation remains vote and competence first. Stable specialization is only a final tie-break and never overrides plurality or competence quality.
