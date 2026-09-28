# Seed81 V4 attempt3 diagnostic protocol

This is a prospective diagnostic repair of attempt2. The V4 Responsibility-only
Layer2 scientific method, GEPA Level-B search, split, seed, model roles, packet,
TeamMiniBatch, Common-Safe, Shadow, and write-back are unchanged. Attempt2 is
permanently `INVALID / NOT_EVALUABLE / DIAGNOSTIC_SAMPLING_INTEGRITY`; its 337
successful provider calls are historical evidence and are excluded from this
sample. Attempt3 requires a new explicit real-API authorization and has no
permission to access Validation50 or Test50.

The sampling unit is `LAYER1_RETURNED_STRICT_POSITIVE_CANDIDATE`: a candidate
in the production GEPA adapter's `LocalOptimizationResult.candidates` with
strict-positive local acceptance evidence. GEPA's internal accepted events and
final frontier are separately recorded as telemetry. They are never used as
the prospective sample or to select a team candidate.

Every returned candidate receives ordinary TeamMiniBatch. Ordinary promotion
admits at most two candidates to ordinary Full. Every unpromoted returned
candidate receives a diagnostic-only Full, which cannot enter Common-Safe,
winner selection, Shadow, or commit. Ordinary Layer2 uses the same returned
candidates, promotion, Common-Safe, winner-only Shadow, and at-most-one atomic
commit as production V4. The diagnostic records full evidence for every
returned candidate without feeding it back into the search.

The target is five returned candidates. At most ten complete opportunities
and twenty reflection proposals are allowed. The GEPA local metric budget is
36 per opportunity. A complete opportunity is never truncated: if the prior
sample has four candidates and the next opportunity returns two, both receive
all diagnostic stages; the observed sample becomes six and the run stops.
With a four-candidate Layer1 return cap, the largest target-crossing sample is
eight. The complete next opportunity is withheld when up to twelve proposals
could exceed the global twenty-proposal ceiling. The stop status reports the
actual final sample size and whether the target was reached.

Conservative per-opportunity call bounds are: local Solver 66 (seed12, metric
limit36 checked between iterations, final iteration at most18), reflection12,
TeamMiniBatch48, mandatory Full400 (ordinary Full at most200 is a subset),
and winner-only Shadow50. Initialization costs at most500 logical Solver
calls. Across ten opportunities the logical provider-call bound is 6,260.
At most 23 transport attempts per logical call gives 143,980 attempts. The
preregistered operational ceilings are 7,000 successful provider calls and
150,000 transport attempts. These are emergency safeguards, not scientific
stopping criteria.

The frozen source SHA, protocol hash, split hashes, dependency identity,
governance bundle, and independent preflight records define attempt3. Any
inconsistency aborts before the provider boundary. The real attempt remains
unauthorized until the user grants a new, attempt3-specific authorization.
