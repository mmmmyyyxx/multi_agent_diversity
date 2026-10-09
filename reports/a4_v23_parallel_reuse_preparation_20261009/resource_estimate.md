# Fresh resource envelope

The initial reconstruction needs at least 11 and at most 32 new
successful semantic Solver responses, before transport failures. Exactly 330
historical physical draws are reusable; 27 later draws are rejected at the first
capacity divergence. No affected case receives a reset retry budget.

For a conditional first-draw-success estimate, the eleven affected requests use
6144; their new input usage is unknown. Expected completion cost is not established.
Initialization's hard reservation bound is calculated from exact wire bytes plus
4096 input margin and the draw's 3600/6144 output cap; transport recovery can take
21 attempts per semantic draw and charges unknown failures in full.

The complete five-opportunity envelope retains 1570 logical Solver observations,
6280 semantic draws, 300 Gradients, five clusters and 30 mutations before reuse.
The new output-only semantic envelope is at most 39,222,630 tokens, excluding
inputs and transport failures. This far exceeds the proposed fresh 2M ceiling.
The runtime therefore may close an incomplete budget-limited Pilot; the allowance
cannot guarantee five opportunities, Full or a commit. No budget increase or
experimental rerun is permitted without a new scope.

Historical charged cost is 763,607. If the proposed scope is approved and fully
spent, these three V2.3 attempts total at most 2,763,607 charged tokens. Reuse is
reported separately from new usage and cannot transfer an old authorization.

## Exact initial request reservation bounds

Reconstructed actual wire bytes give a conservative reservation bound of 124,681
tokens if each of eleven affected profiles succeeds on its first new draw.
All 32 remaining draws bounded by 6144 give 362,314 tokens, excluding transport
retries. With 21 transport attempts per draw, the mathematical reservation envelope
is 7,608,594, which cannot fit the proposed 2M ceiling. The ledger enforces the
ceiling and drains issued calls on failure; no completion guarantee follows.
These are upper bounds, not estimates of the provider's eventual measured bill.
See [machine bounds](resource_bounds.json).
