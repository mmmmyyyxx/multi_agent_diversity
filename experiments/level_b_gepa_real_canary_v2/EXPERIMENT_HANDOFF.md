# Execution handoff: Level-B GEPA canary v2

Status: `PENDING_EXPLICIT_API_AUTHORIZATION`

Do not execute yet. The superseded `reflectionfix1` pending attempt is invalid
and must never be executed. After a new explicit authorization, the scientific owner must
update and re-hash the manifest authorization fact, create a fresh prep identity,
verify the exact execution commit and protocol hash, and hand off only the frozen
command. No source, model, split, budget, GEPA setting, scheduler, or acceptance
change is allowed during that authorization transition.

The future executor may run only one Seed78 parent, one target, and one local GEPA
opportunity. It must preserve the transactional lifecycle, durable ledger,
proposer diagnostics, Validation50=0, Test50=0, no-resume, and no-experiment-retry
rules. Any integrity or infrastructure failure fails closed without retry.
