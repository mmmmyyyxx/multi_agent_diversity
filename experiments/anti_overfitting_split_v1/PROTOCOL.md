# Anti-Overfitting Shadow-Gated Training Protocol v1

This Phase-A freeze changes only split and write-back governance. It adds no method module.

Optimize100 performs all adaptive search. Exactly one train-side Common-Safe winner may be evaluated on Shadow50. Shadow cannot rank, generate, retry, or feed back. A commit requires Optimize Common-Safe and Shadow VoteDelta >= 0, with target-member loss no worse than -2/50. The shadow-gated arm stops after six consecutive opportunities without an approved commit, with an absolute maximum of 32 opportunities. Validation50 is evaluated once after final freeze. Test50 remains inaccessible until a separate final authorization.
