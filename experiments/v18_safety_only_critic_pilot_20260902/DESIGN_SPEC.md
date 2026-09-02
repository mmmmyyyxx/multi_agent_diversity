# Safety-Only Critic Prospective Pilot

This paired fixed-parent pilot isolates one intervention:

```text
Arm A: canonical_llm
Arm B: deterministic_safety_only
```

For each V18 seed 59, 60, and 61, the case selector takes the earliest
historically Critic-blocked Hybrid branch and the earliest historically
Critic-passed Hybrid branch, ordered by update and target. The six cases are
frozen before prospective API execution. No case may be added after readout.

Both arms use the same parent, target, peers, train pool, initial Teacher
request, Student settings, source candidate budget, loss-blind revision rule,
Common-Safe constraint, and ranking. Initial Teacher results are replayed
across arms. Student results are replayed whenever the complete request is
identical; divergent Critic decisions may legitimately create different
downstream Student requests.

The deterministic gate rejects only malformed Teacher plans, explicit
anti-cheating/memorization instructions, and explicit output-contract
contamination. It does not judge specificity, responsibility alignment,
predicted preservation, collateral risk, accuracy, or candidate quality.

All train candidates are evaluated and the hypothetical winner is frozen
before validation. Validation evaluates only frozen hypothetical winners and
never selects a candidate. Test is forbidden. Prompt-team commits and
trajectory mutations are forbidden.

Frozen labels:

- `SAFETY_ONLY_CRITIC_SUPPORTED`
- `THROUGHPUT_ONLY`
- `SEMANTIC_CRITIC_HAS_FILTERING_VALUE`
- `NO_CLEAR_SIGNAL`

