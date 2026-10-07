# Training-process analysis index

1. [Execution and stopping](final_status.json), [stop reconstruction](stop_trace.json), [team states](team_state_trajectory.json).
2. [WHO: all five members](responsibility_trajectory.json), [opportunities](opportunity_summary.csv), [member summary](member_training_summary.csv).
3. [Per-example gradients](gradient_trajectory.json), [all Patterns / same-F / representatives / preservation / transition](pattern_trajectory.json).
4. [Memory actual reads, writes, updates and evictions](memory_trajectory.json). Exact states are restorable from the local bundle after every generation and opportunity.
5. [All candidate generations](candidate_trajectory.jsonl), [pool and gate funnel](candidate_funnel.json), [prompt lineage](prompt_lineage.json), [atomic transitions](transition_trace.json).
6. [Provider usage](provider_usage.json), [charge timeline](cost_trajectory.json), [ledger audit](accounting_integrity_audit.json), [scope audit](governance_audit.json).

An absent stage means NOT_REACHED. Generated contract failure is retained as incomplete execution and does not authorize guard changes, more than three draws or an efficacy-driven rerun. Exact-text bundle location and integrity hash are recorded in final_status.json. No Shadow/Validation/Test sample data are copied into that bundle.

7. [Owner scientific completion receipt](owner_scientific_completion_receipt.json) preserves the aborted runner status separately from scientific validity; [post-stop repair verification](post_stop_engineering_verification.json) records the zero-API engineering correction and current-suite results.
