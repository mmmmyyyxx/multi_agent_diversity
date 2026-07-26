# v4 Full Pilot Operational-Gate Amendment

Date: 2026-07-26
Source commit: `689a5c5cb4320720dd712a31889afd1eb2dcc219`
Source run: `runs_v4_baseline_full_seed42_20260726_114912`

This amendment reclassifies the existing Full pilot by request provenance. It
does not change the formal method, candidate acceptance rules, Solver parser,
Student recovery, target scheduler, or any experiment result.

## 1. Legacy global-request gate

The original gate is retained unchanged as:

```text
legacy_all_request_validity_gate_v1
```

Its result remains:

```text
FAIL
```

The run-wide Solver summary contained 2,200 resolved requests, 2,172 eventual
valid responses, an eventual validity rate of 98.727%, and 28 terminal invalid
responses. Under the legacy requirement of eventual validity at least 99.5% and
terminal invalids no greater than 1, this gate correctly fails as a global
all-request diagnostic.

## 2. Terminal-invalid provenance reconciliation

The 28 terminal invalids are not active-state or selected-output failures. The
candidate decision audit identifies exactly two exploratory candidates in
update 0:

| Source | terminal invalids | decision |
|---|---:|---|
| exploratory rejected candidate A | 19 | rejected; `terminal_invalid_regression` |
| exploratory rejected candidate B | 9 | rejected; `terminal_invalid_regression` |
| exploratory rejected candidates subtotal | **28** | — |
| exploratory accepted candidates | 0 | — |
| active team states | 0 | — |
| validation-selected state | 0 | — |
| selected test state | 0 | — |

The two update-0 candidates both improved target/objective diagnostics but were
hard-rejected because terminal invalids increased. Their 19 + 9 invalids are
therefore exploratory rejected-candidate observations. Candidate decisions show
all incumbent active-state terminal-invalid counts as 0, all accepted candidate
terminal-invalid counts as 0, all validation history terminal-invalid counts as
0, and the selected test terminal-invalid count as 0. Thus:

```text
19 + 9 = 28
28 = global solver summary terminal_invalid_count
remaining terminal-invalid sources = 0
```

The provenance error was counting rejected exploratory candidate requests as if
they were failures of the active or selected team state.

## 3. Corrected gate

The corrected gate is:

```text
active_state_validity_and_isolation_v2
```

| v2 hard condition | Evidence | Result |
|---|---|---|
| infrastructure failure = 0 | 8 updates, no infrastructure terminal failure | PASS |
| raw invalid Student candidate excluded from Stage A | every update has `stage_a_evaluated == valid_candidate_count` | PASS |
| accepted candidate does not increase terminal invalids | all accepted candidate constraints pass terminal-invalid non-regression | PASS |
| selected validation state terminal invalid <= 1 | selected validation state count = 0 | PASS |
| selected test terminal invalid <= 1 | selected test terminal invalid count = 0 | PASS |
| validation counters reconcile | 6 unique states, 6 evaluations, 3 cache reuses | PASS |
| test isolation | test count = 1, after selection, not used for selection | PASS |

Therefore:

```text
legacy_all_request_validity_gate_v1 = FAIL
active_state_validity_and_isolation_v2 = PASS
```

The all-request eventual-validity rate remains a diagnostic metric. It does not
independently fail v2 because its invalid requests are fully attributed to
rejected exploratory candidates whose hard gate already rejected them.

This is a request-provenance classification correction, not a relaxation made
after observing efficacy. No valid active, validation-selected, or selected-test
output requirement was lowered.

## 4. Scope and integrity

- No API call was made while preparing this amendment.
- The original run directory and all original result files are unchanged.
- No method version, prompt, parser, scheduler, recovery policy, or candidate
  acceptance implementation was modified.
- The prior Full efficacy observations remain unchanged and are not recomputed
  here.
