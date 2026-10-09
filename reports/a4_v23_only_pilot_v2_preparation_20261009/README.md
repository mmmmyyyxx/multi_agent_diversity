# Fresh V2.3 operational followup: approval pending

Attempt `a4_v23_only_seed81_20261009_attempt2` is source/startup frozen and has made zero API calls.
The previous attempt ended before optimization because its owner review deadline
expired. This preparation changes supervision, without changing the scientific
method, review timeout, approval guards, models, seed, budget or opportunity cap.

The scientific owner will execute the sole production command and monitor at
intervals of at most 60 seconds. Both actual-evidence reviews remain independent
owner decisions. No prior response cache, profile or unused allowance is reused.

Proposed scope: qwen3-8b V6, thinking=false; qwen3.7-flash Gradient/clustering/
mutation; seed 81; five members; Optimize60; winner-only Shadow40; at most five
opportunities total; at most 2,000,000 new charged+reserved tokens. Validation
and Test calls are zero. No experimental retry or scientific resume is approved.
The cap is not a guarantee of five completed opportunities. The
[resource envelope](resource_estimate.md) remains the prior conservative bound.

The previous attempt already charged 404,677 tokens. Approving the proposed
new ceiling would permit a combined maximum of 2,404,677 across these two scopes.
The old 1,595,323 unused tokens are closed and cannot be transferred.

Source: `f5022a4dc0ccadbf806284b0a003a56c5dec6651`. Startup: `fc02fc0ecdcfa2dd8ad90a46fa08e8ead9bdf58de05120444d6fa0587b04bfc8`.
Exact scope SHA256: `17768e91f01f9ae1a62009ee73858f14b0ee3d6b83ffce1f8ee29b0e94243c18`.

Inspect [freeze](freeze.json), [handoff](execution_handoff.yaml),
[pending authorization](authorization_receipt.json),
[current regression](regression.json) and the
[previous failure report](../a4_v23_only_pilot_execution_20261009/README.md).
READY_TO_RUN stays false until this exact attempt and direct owner execution
receive explicit human approval under AGENTS.md section 8 and task section 9.
