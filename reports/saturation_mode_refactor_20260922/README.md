# Saturation-mode stopping refactor

This zero-provider refactor adds two coexisting execution regimes to all four
unified backend modes: fixed-budget and saturation. Saturation disables normal
scientific caps and stops only after repeated complete units with no accepted
deployable update. High emergency ceilings remain operational aborts and never
mean convergence.

## Complete units

| Mode | Complete local unit | Patience reset |
|---|---|---|
| GEPA_NATIVE | pinned native shuffled-data epoch | strict accepted GEPA mutation |
| GEPA_LAYER2 | full frozen packet schedule pass | strict accepted local mutation |
| MARS_NATIVE | T/C/S plus Target evaluation round | new best deployable prompt |
| MARS_LAYER2 | packet-owned T/C/S plus Target round | strict accepted local mutation |

Layer-2 outer patience is separate and resets only after atomic write-back. A
Vote-neutral Common-Safe commit counts; a local candidate rejected by team
gates does not. Packet exhaustion starts the next deterministic replay epoch
without native sampling fallback.

Recommended pre-freeze options are local patience 2/3/5 and team patience
1/2/3; the initial middle recommendation is local=3 and team=2. These values
must be preregistered before any real saturation run.

Saturation results estimate approximate within-method performance ceilings.
They do not answer equal-budget efficiency and must not be used for unqualified
GEPA-vs-MARS superiority claims.

API calls = 0; Validation50 calls = 0; Test50 calls = 0.
