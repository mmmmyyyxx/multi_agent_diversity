# Historical and concurrency regression

Existing BBH Unified Search, one-opportunity historical BBH replay, and the GEPA concurrent local batch test are included in the focused zero-API suite. No GEPA implementation or configured provider concurrency was changed. The regression test retains `evaluate_batch`, order and stage assertions from the previous implementation.

The broad suite uses the repository's credential-free network guard. Its exact result and any historical artifact exclusions are recorded in `test_summary.json`. Historical Formal V3 source, manifests and run reports were not modified.

The unfiltered guarded run reached `1686 passed / 15 failed / 13 errors / 4 skipped`. Every failure/error was reproduced in these eight files and traced to absent historical private `runs/` inputs; the files also contain passing tests, which are counted only in the unfiltered total:

1. `tests/test_accepted_local_mutation_team_transfer_v2.py` — private bundle absent.
2. `tests/test_common_solver_contract_replay.py` — private replay registries absent.
3. `tests/test_local_gepa_phase_b_v2_freeze.py` — private selected parent tasks absent.
4. `tests/test_sequential_symmetry_breaking_online_pilot_v1.py` — frozen private parent task absent.
5. `tests/test_v16_fixed_parent_generation_probe.py` — historical run metadata absent.
6. `tests/test_v16_generic_m20_fixed_parent_probe.py` — historical run metadata absent.
7. `tests/test_v16_m20_collateral_structure_audit.py` — historical probe summary absent.
8. `tests/test_v16_m20_m2e_fixed_parent_probe.py` — historical run metadata absent.

The guarded broad rerun excludes exactly these eight files; no blanket category or current test is excluded.
