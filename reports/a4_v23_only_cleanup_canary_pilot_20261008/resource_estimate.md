# Resource envelope before paid authorization

The requested ceiling is 2,000,000 charged tokens for the complete continuous
Canary/Pilot attempt. Charges and outstanding reservations share that ceiling.
Missing, invalid or untrusted usage and unresolved reservations receive the
full frozen reservation charge. Concurrency is one; no older balance is used.

There is no measured real V6 cost distribution for this attempt. The earlier
answer-only experiments cannot supply a reliable V6 average-cost forecast.
The frozen capacity envelope materially exceeds 2M and is not a forecast:

| Envelope | Maximum |
|---|---:|
| Logical Solver requests, including initial Optimize and possible initial Shadow | 1,570 |
| Completed Solver draws with four semantic attempts | 6,280 |
| Independent Gradient requests | 300 |
| Cluster requests | 5 |
| Mutation requests | 30 |
| Successful provider responses | 6,615 |
| Transport attempts with the frozen 21-attempt maximum | 138,915 |
| Output accounting ceiling, one Solver draw per logical evaluation, no transport failures | 6,290,310 tokens |
| Output accounting ceiling with four Solver draws | 23,246,310 tokens |

These output-only maxima exclude input and failed-transport reservation charges.
The 1,570 Solver envelope includes all five opportunities reaching their maxima;
cache reuse, fewer wrong examples, `NO_SAFE_EDIT`, rejection and early scientific
stopping can reduce cost. This preparation supplies no empirical completion
probability or expected token total. Initial usage will be audited at Canary.

Authorization at 2M therefore permits a budget-truncated experiment; it does
not guarantee five opportunities or a Full evaluation. Reaching the cap closes
the attempt without increasing the allowance. An incomplete resource-limited
run cannot establish V2.3 method failure. Any later attempt or larger allowance
requires its own scientific scope and exact single-use authorization.
