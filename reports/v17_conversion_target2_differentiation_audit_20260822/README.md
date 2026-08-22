# V17 Conversion Target-2 Differentiation Audit

## Objective

This zero-API audit asks which minimal state-local priority signals can actually
change Hybrid's second target among members that already have legal conversion
responsibility. It does not evaluate candidate quality, validation transfer, or
method efficacy, and it does not change the frozen method.

The motivating five-parent result remains:

```text
Conversion-aware binary pool filtering was not evaluated.
Base Target-2 equaled Conversion-Aware Target-2 for 5/5 parents.
```

The equality occurred because Base Hybrid's RR target was already
conversion-eligible in every prospective state. It is not a zero-effect result.

## Evidence and population

The audit deterministically reconstructs all 24 available V17 S2 historical
states across Seeds 56, 57, and 58. Fourteen contain at least one conversion
residual (`0 < G <= H`). The previously frozen five-parent prospective subset is
reported separately.

Only train-state information available before candidate generation is used:

- frozen responsibility assignments;
- current `G/H` structure;
- existing responsibility repair lane;
- current dominant-wrong coalition membership;
- deterministic RR order.

Candidate outcomes, validation, test, API calls, historical success, and
`WOULD_COMMIT` are not used.

## Minimal signals

Each diagnostic selector maximizes exactly one integer count among the remaining
responsibility-eligible members and uses the unchanged RR order only as a tie
break. These are analysis rules, not proposed method settings.

1. `conversion_responsibility_count`: number of current conversion residuals
   legally assigned to the member.
2. `singleton_conversion_count`: assigned conversion residuals with `G=1`.
3. `direct_vote_flip_count`: assigned conversion residuals already classified
   by the frozen responsibility system as `direct_flip`.
4. `dominant_wrong_weakening_count`: assigned conversion residuals where the
   member currently belongs to a largest wrong-answer coalition.

No weights, learned scores, persistent state, outcome information, or API calls
are introduced.

## Differentiation results

| Signal | All states | Conversion-active states | Five-parent subset |
| --- | ---: | ---: | ---: |
| Conversion responsibility count | 9/24 | 9/14 | 3/5 |
| Singleton (`G=1`) count | 7/24 | 7/14 | 2/5 |
| Direct-vote-flip count | 7/24 | 7/14 | 3/5 |
| Dominant-wrong weakening count | 9/24 | 9/14 | 3/5 |

These counts are Target-2 choices different from Base Hybrid.

The binary conversion filter changed 0/5 choices, whereas count-based signals
create nonzero, non-singleton intervention support. However, the signals are not
equivalent: on the five-parent subset, conversion-count and direct-flip rules
select different members at 2/5 parents, and direct-flip and singleton rules
differ at 3/5.

## Dominant-wrong redundancy

For every member in all 24 reconstructed states,
`dominant_wrong_weakening_count` exactly equals
`conversion_responsibility_count`. Consequently it reproduces the same Target-2
choice at every state and adds no new discrimination in this evidence pool.

This does not prove the signals are mathematically identical in other datasets;
it shows that dominant-wrong membership cannot isolate a distinct next pilot
using the currently available V17 states.

## Five-parent details

| Parent | Base | Conversion count | Singleton count | Direct flip count |
| --- | ---: | ---: | ---: | ---: |
| Seed56 update 4 | 4 | 0 | 4 | 0 |
| Seed56 update 7 | 0 | 4 | 4 | 1 |
| Seed57 update 6 | 0 | 3 | 3 | 1 |
| Seed57 update 7 | 1 | 1 | 1 | 1 |
| Seed58 update 6 | 1 | 1 | 1 | 1 |

The first three parents provide intervention support; the last two retain Base
through RR tie-breaking.

## Interpretation

The result updates the mechanism chain to:

```text
W1 allocation failure
  -> Hybrid restores throughput
  -> recovered coverage remains singleton
  -> binary conversion filtering creates no intervention
  -> conversion opportunity counts can create distinct Target-2 choices
```

It does not establish which changed choice is useful. In particular:

- breadth (`conversion_responsibility_count`) and immediate vote potential
  (`direct_vote_flip_count`) both differ from Base at 3/5 prospective parents,
  but disagree with each other at two parents;
- singleton depth changes 2/5 choices and targets a different causal hypothesis;
- dominant-wrong weakening is redundant with breadth in this dataset.

Therefore a future API experiment must preregister one explicit hypothesis (or
a small causal comparison) before observing candidate outcomes. This audit does
not promote any signal, change Hybrid, or authorize an API run.

## Integrity

```text
API calls = 0
validation calls = 0
test calls = 0
method changes = 0
prompt commits = 0
trajectory mutations = 0
```
