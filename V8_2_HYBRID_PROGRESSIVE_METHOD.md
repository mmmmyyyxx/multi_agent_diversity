# V8.2 Hybrid Progressive Prompt Optimization

V8.2 is an opt-in extension selected by:

```text
shared_vote_tcs_competence_depth2_progressive_residual_hybrid
```

It does not alter the historical V7 or V8 settings. The method version is `v8_2_hybrid_progressive`; its schedule, selector, beam, and TCS policy versions are written to run metadata, checkpoints, histories, selected state, and accuracy exports.

## Training Principle

Boundary examples are high-value but low-support signals. Early optimization must first create individual competence and coverage depth before boundary optimization can work reliably. Margin targeting is therefore progressive rather than exclusive, and competence remains a residual objective even in the late phase.

The fixed optimization-split probe supplies schedule snapshots. Epoch 1 uses `s=0`; later strengths depend on bottom-2 gain relative to the initial prompts and are gated by mean accuracy, C1, and C2 preservation. Validation and test never control the schedule.

## Hybrid Selection

Each update scores agents from normalized individual error, probe weakness, C1 and C2 creation opportunities, actual plurality counterfactual fixes, dominant-wrong redundancy, shared residual error, and capability-gap affinity. At strength `s`, global error and C1 remain positive while boundary and residual weights rise.

Diagnostic evidence remains inside the existing generation budget. It distinguishes general errors, C1/C2 creation, actual plurality-boundary cases, and residual/shared errors. Missing boundary cases do not stop optimization.

## Reward And Guards

The competence component separately rewards and penalizes C1 and C2 transitions. Its blend weight is `max(0.30, 1-s)`, so competence pressure never disappears. The four Pareto objectives remain vote gain, vote loss, target accuracy, and one stage auxiliary objective.

Invalid output, incomplete or overlong prompts, invalid-rate regression, C1 net loss, and target-accuracy loss beyond 0.05 are hard failures. Mild error dependence, residual cycles, mechanism shifts, and mild accuracy regression are soft penalties applied to beam ranking.

## Safe, Exploit, Explore

A three-item beam retains the safe incumbent, a hard-feasible Pareto exploitation candidate, and a controlled exploration candidate when available. Exploration requires nonnegative competence evidence and a real ordered mechanism-signature change. It is retained for future search but never becomes active merely because it occupies the explore slot.

TCS asks Student for two candidates per parent: `task_specific_repair` and `mechanism_alternative`. Both declare ordered `mechanism_steps`, target failure buckets, and expected effect. The mechanism signature is normalized locally without an LLM call; persona changes, synonyms, and step renumbering do not establish mechanism novelty.

## Selection And Reporting

Validation uses `vote_generalization_first`: plurality vote accuracy, mean individual accuracy, bottom-2 accuracy, C1, C2, normalized plurality margin, invalid rate, then earlier epoch. This avoids rewarding an apparently flatter team when its mean competence is lower.

Report aggregation-gap reduction only together with C1. A smaller gap is oracle-preserving only when final C1 remains within the comparison tolerance of baseline. `shared_guarded_beam` uses a reused-file protocol and is not a strict same-protocol causal comparison for V8.2.
