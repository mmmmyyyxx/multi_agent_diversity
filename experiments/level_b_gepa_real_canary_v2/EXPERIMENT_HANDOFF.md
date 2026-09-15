# Execution handoff: Level-B GEPA canary v2

Status: `AUTHORIZED_FOR_ONE_FRESH_EXECUTION`

The superseded `reflectionfix1` pending attempt is invalid and must never be
executed. The user explicitly authorized the `precallclosure1` attempt on
2026-09-15. Re-hash the authorization fact, create a fresh authorized prep
identity, verify the exact execution commit and protocol hash, and execute only
the frozen command. No source, model, split, budget, GEPA setting, scheduler, or
acceptance change is allowed during the authorization transition.

The future executor may run only one Seed78 parent, one target, and one local GEPA
opportunity. It must preserve the transactional lifecycle, durable ledger,
proposer diagnostics, Validation50=0, Test50=0, no-resume, and no-experiment-retry
rules. Any integrity or infrastructure failure fails closed without retry.
