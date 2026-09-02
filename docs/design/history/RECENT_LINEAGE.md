# Recent Design Lineage

This compact index mirrors `experiments/registry.yaml` and existing reports. It
does not replace those evidence artifacts or reinterpret their metrics.

| Node | Parent | Scientific question / sole delta | Frozen facts | Result label | Evidence | Implementation / result commit | Status |
|---|---|---|---|---|---|---|---|
| V17 formal | unknown | Five-arm formal comparison | task, splits, aggregation | `MULTI_CONTRAST_MIXED` | `reports/v17_formal_5arm_3seed_20260813` | `76e6960` / `ef9124e` | completed |
| V17 failure decomposition | V17 formal | Decompose S1-to-S2 degradation offline | trajectories and recorded outcomes | `TARGET_CONCENTRATION_ASSOCIATED_WITH_MEMBER_TRANSFER_REGRESSION` | `reports/v17_failure_decomposition_20260820` | unknown / `6b739c0` | completed |
| 2x2 target-allocation isolation | V17 failure decomposition | Cross allocation with residual context | fixed parents, peers, rollout | `TARGET_ALLOCATION_DOMINANT` | `reports/v17_module1_2x2_causal_isolation_20260820` | unknown / `2016387` | completed |
| Hybrid selector | 2x2 isolation | Change only second target allocation | first target and proposal pipeline | `HYBRID_THROUGHPUT_ONLY` | `reports/v17_hybrid_target_allocation_pilot_20260821` | `43230a5` / `30e42e0` | completed |
| V18 online accumulation | Hybrid selector | Run matched W1/Hybrid trajectories | horizon and proposal pipeline | `LONGITUDINAL_ACCUMULATION_WITH_VOTE_CONVERSION` | `reports/v18_hybrid_online_accumulation_pilot_20260822` | `a7032b6` / `5cb3724` | completed |
| Write-back quality audit | V18 online | Decompose accepted-pool quality | recorded candidates and ranking | `COMMON_SAFE_FEASIBLE_SET_QUALITY_GAP_WITH_EXISTING_TRAIN_VOTE_LOSS_RISK_SIGNAL` | `reports/v18_writeback_quality_diagnostic_20260824` | unknown / `f6fdc0f` | completed |
| M2F trigger extension | Write-back audit | Extend one compatibility trigger | parents, acceptance, ranking | `EXTENDED_M2F_TRIGGER_NOT_SUPPORTED` | `reports/v18_m2f_trigger_extension_pilot_20260824` | unknown / `a00484c` | completed |
| GEPA candidate-selection audit | Write-back audit | Replay frontier; attempt N=2 vs N=4 | target, parent, Common-Safe | `CANDIDATE_SELECTION_NOT_PRIMARY__BREADTH_NOT_EVALUATED` | `reports/gepa_candidate_breadth_audit_20260831` | unknown / `9a21dfe` | completed |
| Critic gate audit | GEPA audit | Locate pre-candidate stoppage | historical branch outcomes | `PRE_STUDENT_CRITIC_GATE_BOTTLENECK_CONFIRMED` | `reports/gepa_critic_gate_failure_audit_20260831` | unknown / `400a841` | completed |
| Safety-only counterfactual | Critic gate audit | Remap historical rejection categories | historical decisions | `SAFETY_ONLY_REACH_BOUNDS_ONLY` | `reports/v18_critic_safety_only_counterfactual_audit_20260902` | unknown / `944aaec` | completed |
| Safety-only prospective | Counterfactual | Replace Critic with deterministic gate locally | fixed parent, Student, rollout | `NO_CLEAR_SIGNAL` | `reports/v18_safety_only_critic_pilot_20260902` | `ac34bda` / `5c33264` | completed |
| Historical Teacher safety | Safety pilot + V18 online | Audit field-local hard-safety markers | historical plans and outcomes | `STABLE_PATTERN_NOT_DISCRIMINATIVE_FOR_CANONICAL_BLOCKING` | `reports/v18_historical_teacher_safety_failure_audit_20260902` | unknown / `f61e14f` | completed |
| Shadow-Raw pilot | Critic gate + Teacher audit | Continue after unchanged canonical rejection | same plan, feedback, parent | `CRITIC_OVER_FILTERING_CAUSALLY_SUPPORTED` | `reports/v18_shadow_raw_critic_pilot_20260902` | `c4542cc` / `474b85d` | completed |
| Teacher-Critic four-arm | Shadow-Raw + Teacher audit | Cross Teacher-Clean with Critic role | fixed parents, Student, rollout | `C_NO_SEMANTIC_CRITIC` | `reports/v18_teacher_critic_pipeline_ablation_20260902` | `14f7fbd` / `ec2965a` | selected for next online validation |
