# V18 accepted-candidate counterfactual ranking audit

This zero-API audit inventories every feasible candidate in each accepted C
update for Seeds68-70. It first checks whether unselected feasible candidates
have complete frozen validation-cache coverage. Validation outcomes are compared
only when they were already observed; missing outcomes remain unknown.

The train-side audit compares target gain, Vote gains/losses/net, coverage and
dominant-wrong proxies, frozen ranks, and winner status. It does not use
validation to rerank candidates and does not treat train-side proxies as
counterfactual validation outcomes.

The audit may conclude that ranking is unsupported by available evidence, that
some commits necessarily reflect feasible-set quality because no alternative
exists, or that the global distinction is unidentifiable. It may not call an
API, access test, modify the method, or rewrite historical artifacts.
