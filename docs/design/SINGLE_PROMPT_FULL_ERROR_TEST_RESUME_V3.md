# A2 paid Test continuation V3

This independent Test-only amendment continues the three completed A2 cells from
`single_prompt_full_error_control_v2`. It does not train, select Prompts, access
Validation, run a Teacher, or change production scoring. The original source and
all original run artifacts remain immutable. A new source/scope and exact human
authorization are required before any new provider dispatch.

## Observed failure and scoped scoring repair

The original Test stopped at a local mathematical verification timeout. Offline
reproduction of its saved answer also exceeds eight seconds in the native worker;
the existing exact power-certificate worker finishes the same pair in less than
one second, with a certified inequality. This reuses the already tested V15
certificate implementation, without approximating equality or adding an item
exception. The native parser, mathematical settings and native predicate remain
the fallback for unsupported or near/equal cases. Only this continuation uses the
existing power worker and its thirty-second process deadline. Production and
historical source snapshots are unchanged.

Local parsing and scoring have sixteen admission slots. Each timeout retries the
same response locally, never generates a replacement paid answer and never becomes
an incorrect score. After three same-context timeouts, retain the worker's private
trace and fail with `MATH_LOCAL_SCORING_STALLED` for diagnosis; identity, receipt,
authorization, persistence and other non-network failures also fail closed.

## Exact evidence continuation

The original Test ledger contains359 physical attempts:350 successful paid draws
and9 typed network failures,523754 accounted Tokens. Training contains33989
attempts and86507149 Tokens. Network conservative charges are included in those
totals, not inferred supplier usage. Original training and Test cost therefore
starts at34348 attempts and87030903 Tokens. No ancestor cost is charged again.

Audit all sixty training transitions using adjacent checkpoints, the complete
sealed inventory, full ledgers and receipts. Retain only compact final Prompt
records for execution. Freeze the complete parent attempt inventory. The runner
performs this semantic audit once at startup, then rehashes all parent bytes and
source identity at cell boundaries and closure; no persistent audit cache exists.

Reuse all350 successful draws only when cell, stage and complete serialized
provider request match. Preserve the original request namespace. Network failures
are not successful responses and are not replayed. Reconstruct partial-cell
observations using these paid draws and require every recorded terminal observation
hash to match exactly. Changed requests, cross-repeat draws, duplicate replay or
prior-score drift fail before acceptance. New requests retain the original
gpt-4o-mini model, decoding/output capacities, SDK2.37.0, Seed42, Test300 members
and repeat/phase cache isolation. No Optimizer, Gradient, Cluster or Memory calls
are permitted. There are three final Prompts and900 terminal Test observations.

TrainingSolver settings are inherited metadata only. Actual TestSolver96 and
local admission16 remain unchanged. The separate inherited350 plus new successful
semantic/inflight attempts cannot exceed3600, so the new ceiling is3250. Failed
network physical attempts and cumulative Tokens have no hard cap. Typed connection
and API timeout failures use the original accounted2/4/8/16/30/60 then60-second
backoff indefinitely. Local score retries do not consume Solver semantic draws.

## Reporting

Report Test and original Validation for all three repeats, the original update,
rejection, exhaustion and no-op counts, inherited and new role/stage calls and
Tokens,350 reused draws, preserved terminal observations, local/network retries,
and conservative network charges separately. Do not present a paid response as a
completed score. Do not change original terminal results or call this a new blind
Test: historical Test exposure and the explicitly scoped scoring backend remain
limitations of comparison with prior C2 results. No new training is authorized.
