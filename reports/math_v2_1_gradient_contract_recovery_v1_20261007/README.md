# Bounded Gradient contract recovery V1

Verified sampling amendment: at most three identical-request draws per wrong
example, accepting the first output satisfying the unchanged Gradient contract.
Prompt V3 and Guard V6 are byte-identical. A weak valid output is accepted.
Three invalid outputs stop extraction; infrastructure errors never trigger
contract retries. Exactly one accepted Gradient per wrong example enters the
unchanged clustering and same-F path.

Full current suite: **1495 passed, 2 skipped**.
Historical private tests are NOT_RUN; full historical replay PASS is not claimed.
The real attempt4 rejection remains strong provenance in zero-API replay; a
second synthetic valid draw is accepted without exposing the rejected output.
1005 durable journal cycles and first-valid, exhaustion, infrastructure and
request-equivalence boundaries pass.

See [policy](policy.json), [scientific delta](scientific_delta.json),
[attempt4 replay](attempt4_zero_api_replay.json), [request equivalence](provider_request_equivalence.json),
[accounting bounds](accounting_bounds.json) and [verification](verification.json).
Physical Gradient ceilings increase by three; other role, opportunity, Memory,
Layer1, team-gate and scientific-stop settings remain unchanged. The durable
40M ceiling protects every physical reservation and remains an upper limit.

Start: `05e60bdbb73f2a89ed05ec88db9f4e1fe5a7e579`. Attempt5 is already registered
by the prior closed task, so the new fresh execution uses attempt6. API dispatch
requires the exact source, startup, scope and budget freeze under the new task.
This verification packet contains zero real calls and no efficacy estimate.
