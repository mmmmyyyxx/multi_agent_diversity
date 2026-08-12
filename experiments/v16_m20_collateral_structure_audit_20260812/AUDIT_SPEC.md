# M20 Collateral Structure Audit

This is a read-only, zero-API follow-up to the preregistered Generic-vs-M20
fixed-parent probe. It uses the same eight frozen parents and only the M20
candidate evaluations from the protocol-gate-PASS retry execution. It does not
generate candidates, change prompts, read validation/test data, or reinterpret
the Generic-vs-M20 classifier.

For candidate `c`, `G` contains target-member examples that change from wrong
to correct and `L` contains examples that change from correct to wrong.
Responsibility membership is the case's frozen assigned-question-hash set, so
the primary quantities are `|G intersect R|` and `|L outside R|`.

Every parent-correct example has exactly one competence-retention role:

1. `unique`: the target is the only correct member (`G = 1`);
2. `pivotal`: excluding the target leaves a non-positive peer plurality margin;
3. `stable`: neither above and present in the target's frozen stable-correct set;
4. `fragile`: all other parent-correct examples.

The audit reports candidate-level counts and hashed-example rows. No prompt,
question, gold/model answer, raw response, endpoint, credential, or cache
content is publishable.

Descriptive collateral classes are frozen before inspecting per-example loss
structure:

- `NONE`: no target regression;
- `LOCAL_ACCIDENTAL`: one non-responsibility loss and no unique/pivotal loss;
- `SPECIALIZATION_OVERWRITE`: any unique or pivotal non-responsibility loss;
- `BROAD_DEGRADATION`: at least three non-responsibility losses and losses are
  at least twice responsibility gains;
- `SKILL_TRADEOFF`: remaining regression candidates with responsibility gain;
- `UNEXPLAINED_REGRESSION`: remaining regression candidates without
  responsibility gain.

`SPECIALIZATION_OVERWRITE` takes precedence over `BROAD_DEGRADATION`, which
takes precedence over `LOCAL_ACCIDENTAL`, `SKILL_TRADEOFF`, and
`UNEXPLAINED_REGRESSION`. These are descriptive mechanism labels, not a new
promotion rule.
