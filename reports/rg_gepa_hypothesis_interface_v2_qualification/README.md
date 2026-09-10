# RG-GEPA Hypothesis Interface V2 qualification

This zero-API freeze replaces model-authored JSON keys with one exact four-position array of opaque enum IDs. Array position supplies the program-owned field meaning; the existing deterministic renderer expands only program-owned clauses. No cleanup, alias mapping, fallback, or schema retry is allowed.

The qualification consists of 12 qwen3.7-flash requests over synthetic symbolic contexts. It calls no Solver and uses no experiment parent, dataset row, Validation, or Test evidence. PASS requires 12/12 strictly valid outputs. This engineering qualification is not scientific evidence and does not authorize a fixed-parent retry.
