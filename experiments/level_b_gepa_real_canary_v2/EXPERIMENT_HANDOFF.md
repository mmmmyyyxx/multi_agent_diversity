# Execution handoff: Level-B GEPA canary v2

Status: `PENDING_NEW_EXPLICIT_AUTHORIZATION`

The `precallclosure1` attempt is preserved as an aborted, non-resumable run. Its
Local GEPA evidence remains valid, but its full path is incomplete. The fresh
`stagefix2` attempt closes only the Solver stage-attribution contract. The
unused `stagefix1` prep is superseded and must never be executed. `stagefix2` is not
authorized to call real APIs. A future explicit authorization must create a new
authorized prep identity and fresh formal run root while retaining this exact
source/protocol freeze. No source, model, split, budget, GEPA setting, scheduler,
or acceptance change is allowed during that authorization transition.

The future executor may run only one Seed78 parent, one target, and one local GEPA
opportunity. It must preserve the transactional lifecycle, durable ledger,
proposer diagnostics, Validation50=0, Test50=0, no-resume, and no-experiment-retry
rules. Any integrity or infrastructure failure fails closed without retry.
