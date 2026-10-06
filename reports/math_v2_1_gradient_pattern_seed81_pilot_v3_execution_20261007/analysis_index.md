# Pilot analysis index

1. [Initial symmetry and natural differentiation](initial_symmetry_and_differentiation.json): identical V1_2 prompts, 300 initial evaluations, five independent lanes, exact equality after each commit.
2. [Responsibility allocation](responsibility_trajectory.json): all members' D/N/C/F, frozen discount, feasibility, target sequence; [member totals](member_training_summary.csv).
3. [Gradients](gradient_trajectory.json): one per selected-member wrong example, lengths, contract outcomes, hashes; exact texts remain in the ignored analysis bundle.
4. [Pattern clustering](pattern_trajectory.json): support aliases, shared/singleton counts, coverage, residuals; clustering sees gradients only.
5. [same-F selection](pattern_trajectory.json): D/N/C/F, selected Pattern, deterministic representatives, preservation and transition evidence.
6. [Memory evolution](memory_trajectory.json): empty initial state; actual reads, writes, promotions, evictions, state hashes and counters.
7. [Prompt mutation lineage](prompt_lineage.json), [exact differentiation](initial_symmetry_and_differentiation.json): proposal/parent hashes and commit equality matrices.
8. [Local candidate outcomes](candidate_trajectory.jsonl), [candidate funnel](candidate_funnel.json): every generation, admissibility, local scoring and export.
9. [TeamProbe](gate_outcomes.json): attempts and frozen promotion passes; per-candidate measurements in candidate_trajectory.jsonl.
10. [Full](gate_outcomes.json): attempts, initial floor and strict-gain passes; per-candidate metrics in candidate_trajectory.jsonl.
11. [Shadow](transition_trace.json), [gate totals](gate_outcomes.json): attempts and pass/commit outcomes. No private sample content is published.
12. [Commits](transition_trace.json), [team-state trajectory](team_state_trajectory.json): atomic parent/child identity and Optimize metrics.
13. [Stop](stop_trace.json), [final disposition](final_status.json): frozen scientific stopper or preserved incomplete contract failure.
14. [Cost](cost_trajectory.json), [provider usage](provider_usage.json), [accounting integrity](accounting_integrity_audit.json), [wire and durability](wire_and_durability_audit.json).

These traces describe one A4/Seed81 search. Scientific efficacy is NOT_CAUSALLY_ESTABLISHED. Validation, Test, other arms/seeds, raw diagnostics and LLM judges were not run. A stage not reached is not a failed gate. Exact local artifacts are indexed in final_status.json and are excluded from publication.
