# GEPA Candidate Breadth Fixed-Parent Pilot

This prospective diagnostic freezes exactly two V18 harmful parents: Seed59
update 3 and Seed61 update 5. Parent, target, peers, train pool,
responsibility-conditioned proposal context, Common-Safe, ranking, model,
temperatures, and loss-blind revision policy remain fixed.

One four-source generation is executed per parent. `N=2` is the nested prefix
containing the first two valid source mutations and their one-per-source
loss-blind revisions; `N=4` contains all four requested source mutations and
their revisions. No candidate is committed and no trajectory state changes.

Train-side decisions are persisted before validation. Validation evaluates
only the frozen N=2 and N=4 WOULD_COMMIT winners. Test is forbidden.

The primary breadth evidence is an N=4-only Common-Safe feasible candidate
with zero train Vote loss or train Vote loss below the N=2 pool minimum.
Validation never defines feasibility, ranking, pool membership, or the label's
candidate-quality premise.

Frozen labels:

- `PROPOSAL_BREADTH_SUPPORTED`
- `PROPOSAL_BREADTH_THROUGHPUT_ONLY`
- `NO_PROPOSAL_BREADTH_SIGNAL`
- `PROPOSAL_BREADTH_HARMFUL`
