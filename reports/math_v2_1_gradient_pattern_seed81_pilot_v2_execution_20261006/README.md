# A4 / Seed81 Pilot attempt2: initial-stage persistence abort

The V5 numeric provenance guard and V3 minimal Gradient prompt were frozen and
executed from `b4ec2d3c5025949af1e8dbcfc43042c3c06ac6e6`. This fresh Pilot used
five identical minimal V1_2 initial prompts, independent member lanes, empty
Memory and a fresh cache. No new Canary was run.

The run aborted with Windows `PermissionError` while atomically replacing the
derived token-ledger snapshot, after the authoritative journal had durably
appended a RESPONSE charge. Initialization completed 263/300 logical evaluations.
There were 291 physical Solver attempts and 290 complete response records. The
last charged response lacks raw-response evidence; it has not been reconstructed
or counted as a completed evaluation. Initial team scores were not frozen, and
no Gradient, cluster, reflection, candidate, opportunity or commit occurred.

The full Pilot is **NOT EVALUABLE** and incomplete. This provides no evidence
about the new guard's fresh compliance, semantic usefulness or method efficacy.
Q1–Q5 remain unobserved in this attempt. Zero commits here are not a convergence
result. Validation/Test/Shadow calls are all zero.

The read-only integrity audit verified the frozen source, hashes, raw inventory,
290 recorded responses, request identities and Solver inputs. It explicitly
retains the one-response terminal evidence gap. All 4,703 historical evidence
files remain unchanged. The one-shot scope is [closed](authorization_closure.json)
with zero authorized retries; no automatic restart or rerun occurred.

The authoritative token journal has 291 reliable RESPONSE charges totalling
**74,607 tokens**. Cumulative charge is **1,929,584**, remaining allowance
**38,070,416**, and unresolved reservations **0**. The stale derived snapshot was
rebuilt from the existing journal without changing any journal event or run file.

A separate zero-API Windows test reproduces WinError 5 when an ordinary Python
read handle remains open on the replacement destination; closing it permits the
same replacement. Monitoring reads are a plausible contributor. The actual
incident lock holder was not established. This is a persistence/read-sharing
failure, not a demonstrated numeric-admissibility failure. The owner monitor now derives
budget totals from the hash-chained append-only journal and avoids live reads
of atomic snapshot/lifecycle files. A zero-API control completed 100 snapshot
replacements while an append-only journal reader stayed open. This removes the
monitor's destination-read contention; it does not identify the incident lock
holder or establish complete OS persistence closure. Current scientific identities
are unchanged. Any new real attempt needs a fresh freeze and exact user
authorization under AGENTS.md §§6/8. The consumed scope cannot be reused.

Pre-execution zero-API checks passed all seven historical Canary attempt3
Gradients and all four historical Pilot attempt1 Gradients, including the fourth
generic mathematical constant. All 22 real-source numeric-copy negative controls
were rejected. The frozen-source current suite passed **1,356 tests**, with
**2 skips** and **1,468 deselected** historical/private cases. This is not an
all-historical-replay claim. The hard 400-character limit, Gradient definition,
Pattern, same-F, Layer1, Memory, models, memberships, budgets and stopping remain.

All public artifacts contain only hashes, counts, categories and aggregate
metrics. Raw questions, gold/model answers, prompts and provider responses stay
in ignored private evidence. Audit provider calls and network attempts are zero.
