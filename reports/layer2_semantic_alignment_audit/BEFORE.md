# Layer-2 semantic alignment: pre-change audit

Status: `SCIENTIFIC_DECISION_REQUIRED`. This is an audit of the current tree, not a new experiment freeze. No API was called and no historical artifact was changed.

| Requested semantic | Current evidence | Finding |
| --- | --- | --- |
| Overlapping raw legal responsibility; no historical routing before D/N/C | `team_search/system_runtime.py::freeze_current_responsibility` calls `compute_repair_eligibility_sets` directly. `tests/test_layer2_raw_responsibility.py` checks overlapping assignments and routing-policy poisoning. | Implemented for this path. |
| Primary is target selection, not exclusive ownership | `primary_responsibility_scheduler.py` scores each member's raw assigned opportunities. The overlapping-assignment test retains the same residual for two members. | Implemented in allocation; evidence nomenclature still needs alignment. |
| No independent coalition evidence group | `system_runtime.py::SystemResponsibilityAssignmentFactory` classifies an unassigned wrong residual as `coalition`; `schemas.py::TeamEvidenceCase` and `task_builder.py` admit and consume that group. Direct packet fixtures use coalition rows for local evaluation. | Conflict. |
| TeamMiniBatch12 = 4 repair + 4 preservation + 4 team-hard residual | `task_builder.py::TeamMiniBatchQuota`, `team_search/protocol.py::TeamSearchContract`, `versions.py::TEAM_MINIBATCH_CONTRACT_VERSION`, `CURRENT_SPEC.md::INV-TEAM-MINIBATCH-001`, and `invariants.yaml` all freeze 4 responsibility + 4 coalition + 4 preservation. Seed78's frozen protocol explicitly records that composition. | Existing protocol conflict; a replacement changes experiment meaning. |
| Identical five-member initialization | `config.py` defaults to `shared_identical`. The existing real-state test shows all five members can be legally responsible for the same wrong rows, leaving no old coalition quota. | Baseline is retained, but old quota can fail at initialization. |
| Seed-independent score tie-break | `select_primary_responsibility_targets` uses seed, update index, and RR cursor in the equal-score order. | Conflict; changing it alters the frozen scheduler policy. |
| Layer-1 uses only immutable Layer-2 packet | Packet construction is in `Layer2EvidenceRequestBuilder`; backend isolation and a same-packet/different-global-dataset test remain to be audited or added after the protocol decision. | Not certified by this preliminary audit. |

The existing `tests/test_layer2_raw_responsibility.py::test_shared_identical_parent_exposes_unmodified_coalition_quota_blocker` explicitly expects the old strict quota to fail with `4 unique coalition examples`. Changing that behavior and the tie-break in place would invalidate existing contract identities and alter the meaning of frozen protocols.

Before implementation, decide whether the new semantics become a versioned opt-in Layer-2/TeamMiniBatch/scheduler contract while historical frozen V1 protocols remain intact. The term “currently correct but vulnerable to prompt mutation” also needs an operational preservation-selection criterion; the current code labels all target-correct rows without a vulnerability test.
