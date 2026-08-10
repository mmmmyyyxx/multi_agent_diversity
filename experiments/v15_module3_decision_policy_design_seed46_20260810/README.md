# v15 Module 3 Offline Decision-Policy Design

- Status: **COMPLETE**
- API calls: **0**
- Exact S2/S3 replay validation: **PASS**
- Recommended proposal: **M3_B**

This directory compares B0, B1, M3-A, M3-B, and M3-C on the frozen Qwen3-14B
Seed46 v14 S3 Stage-B pool. Every update retains its actual parent, target pair,
candidate pool, and rollout. Hypothetical winners are never propagated.

The three proposals isolate:

- M3-A: feasibility/progress semantics;
- M3-B: responsibility evidence as secondary selection;
- M3-C: explicit plurality-boundary-aware lexicographic ranking.

No API, validation, test rerun, candidate generation, or new rollout occurred.
No formal method code was changed. The final test was excluded from policy
selection. See `recommendation.md` for the design-only conclusion.
