# Multi-model Solver Headroom Screening — Seed65 Amendment

This amendment supersedes the unfinished three-seed screening after the user
requested broader model coverage and one seed only. The completed
`qwen3-8b` Seed65 Static/Generic pair is retained; its interrupted Seed66 root
is excluded, never resumed, and never analyzed. Seed67 was not started.

## Candidate order

1. `qwen-turbo`
2. `qwen3-4b`
3. `qwen-flash`
4. `qwen3-8b` (completed anchor)
5. `deepseek-r1-distill-qwen-7b`
6. `deepseek-r1-distill-llama-8b`
7. `deepseek-r1-distill-qwen-1.5b`
8. `glm-4.5-air`

Only models visible to the credential and passing one minimal real API smoke
enter the Static probe. Unavailable models are skipped in order. No denied
model is repeatedly called.

## Stage 1: Static-first probe

Every entrant runs exactly Seed65, five agents, canonical shared prompt, train
75, and no optimization updates. A separate read-only evaluator measures the
50-row validation split. Test is never loaded or evaluated.

A Solver qualifies for Generic when all are true:

```text
0.50 <= Static Validation VoteAcc <= 0.64
Static Validation OracleAcc - VoteAcc >= 0.08
terminal-invalid rate <= 0.01
infrastructure failures = 0
```

At most three qualifiers advance, ranked by larger Oracle-Vote gap, then lower
Static VoteAcc, then frozen candidate priority. This selection uses Static only
and is frozen before any new Generic result.

## Stage 2: one-seed Generic

Selected models run canonical Generic with Seed65, 32 updates, one branch, two
candidates, Stage B budget 2, Proposal Memory off, and unchanged Common-Safe.
Their completed Static state and immutable comparison cache are reused; Static
is not rerun. The existing qwen3-8b Seed65 Generic result is reused if selected
or reported as an anchor otherwise.

Evaluator, Teacher, Critic, and Prompt Optimizer are fixed to
`qwen3.7-flash`. All Solver requests use thinking disabled. The final active
state is evaluated once on validation after training; validation never selects
or writes back a state.

## One-seed headroom label

For each Generic-evaluated model, report but do not overclaim:

```text
VoteDelta = Generic Validation VoteAcc - Static Validation VoteAcc
SUPPORTED_LOCAL if VoteDelta >= 0.04 and Generic Oracle-Vote gap >= 0.08
NO_LOCAL_HEADROOM_SIGNAL otherwise
```

This single-seed amendment is a screening diagnostic, not a formal multi-seed
model-selection claim.

Prohibited: Full, Module1, M20, M2F, Hybrid, test access, data changes, extra
seeds, result-conditioned retry, and modification of Generic or Common-Safe.

```text
FULL_METHOD_NOT_RUN=true
TEST_ACCESSED=false
```

## Engineering-only validation retry amendment

The first Static validation attempt stopped before any validation API call
because the evaluator could not restore a routing-disabled Static checkpoint.
That failed output directory is immutable evidence.  After the seed-agnostic
checkpoint reconstruction fix passed its offline regression suite, validation
is authorized exactly once in a fresh `validation_retry1` root.  Existing
Static rollouts are read-only and are not rerun.  Any selected non-anchor
Generic arm uses a fresh `generic_retry1` root initialized from a copy of its
completed Static Solver cache; the `qwen3-8b` Static and Generic anchors remain
read-only.  This amendment changes execution plumbing only and does not change
the frozen model gate, Generic method, seed, data, budgets, or metrics.

## Static gate v2 structural amendment

Before any new Generic training, a zero-API row-level audit established that
all five Static members use one identical prompt hash and produce identical
outputs on all 50 validation rows. Consequently Static Oracle accuracy equals
Static Vote accuracy by construction, so the original Static Oracle-minus-Vote
prefilter cannot distinguish Solver models. With explicit user approval, that
single structurally impossible prefilter is removed. The Static Vote range
`0.50 <= VoteAcc <= 0.64`, terminal-invalid threshold `<= 0.01`, at-most-three
limit, and original ordering by gap (a constant tie), lower Static Vote, then
frozen priority remain unchanged. Generic Oracle-minus-Vote `>= 0.08` remains
a final headroom criterion. This transparent amendment was frozen before any
new model's Generic result existed; existing Static validation and the
qwen3-8b anchor remain read-only.

The first post-amendment Generic launcher (`generic_retry2`) stopped in
preflight before any model call because a runner-only cache argument was sent
to the preflight CLI. That directory is preserved as invalidated engineering
evidence. The behavior-preserving launcher fix is frozen and uses fresh
`generic_retry3` and `validation_retry3` roots; no arm, model, budget, gate, or
metric changed.
