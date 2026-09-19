# local_gepa_parent_acquisition_v1

Phase A is an explicit amendment to the original zero-API preparation restriction
against acquiring additional parents. The original preparation remains
PARENT_CATALOG_INSUFFICIENT; its artifacts are not rewritten. This acquisition
is preregistered before any acceptance-rate pilot proposal outcome exists.

## Authorized acquisition

One fresh shared-P0 initialization on the existing Optimize100 fold A+B, seed 78.
P0 SHA256: 549bc93c03f703faf5aa1bd56b557135fb6e65d0cf6c055b8fad6a15e7c87a63.
Optimize CSV SHA256: d308dcf239d42e949775fff2b8fe7540add825e20c767dee789955e7243b2b45.
The same exact request realization is shared by all five identical members.
There are exactly 100 distinct planned requests, at most 100 successful Solver
requests, and at most 400 transport attempts. No cached historical predictions
are consumed. COMMON_SOLVER_CONTRACT_V1 freezes qwen3-8b, thinking disabled,
temperature 0, max_tokens 1800, timeout 120 seconds, at most four attempts per
request, status-code allowlist and backoffs 1/2/4 seconds. SDK retries are disabled.
Strict parser failure is scored wrong without semantic retry. Truncation,
missing usage, unexpected request identity, persistence failure, exhausted retry,
source mismatch or budget violation terminates acquisition. No resume or second
acquisition is permitted under this authorization. Failed-attempt token usage
may be unknown and is disclosed separately. The successful response output-token
envelope is 180000; request count is the binding resource budget.

GEPA proposals, Reflection, accepted mutations, optimization search,
TeamMiniBatch/Full/Common-Safe/Shadow evaluation, write-back, persistent
realizability updates, Validation50 and Test50 calls are all zero.

## Frozen eligibility and task construction

Only the 100 baseline observations are used. Current production vote state,
counterfactual eligibility, service routing and primary-lane summaries reconstruct
one source state. Empty zeroed responsibility bookkeeping is ephemeral;
persistent realizability is neither constructed nor updated. Explicitly build
each of the five members, without using scheduler rank or baseline accuracy to
choose a member. No optimizer engine is instantiated.

Use the current assignment factory and LocalTaskBuilder unchanged. Their shared
4/4/4 quota helper selects local optimizer validation data only; this does not
invoke TeamMiniBatch evaluation. Search and local validation are Optimize-only
and may overlap as specified by the current local problem; this is not a new
independent held-out split. Complete private LocalOptimizationTask payloads use
seed 78, update zero, budget max_metric_calls=205, reflection_minibatch_size=3,
max_returned_candidates=4. These fields freeze task identity, not authorization
to run an optimizer. Phase B must separately preregister and authorize its budget.

An eligible task requires a current-contract-valid decision procedure, explicit
target and non-fallback primary lane, nonempty assigned residuals, complete
search payloads and twelve local validation rows (four each responsibility,
coalition and preservation), unique IDs, exact Optimize payload matches,
reconstructable context, exact serialization roundtrip, current Solver/output
contracts and absent backend state. Freeze ordered IDs, payload hashes, prompt
hash, context hash, source state and task identity. Reject duplicate local
problem content. No post-hoc missing-field completion is allowed. Integrity
corruption aborts; insufficient data quotas exclude the affected task.

Enumerate every eligible subset of four. Maximize lexicographically distinct
source states, distinct targets, then distinct primary lanes. Break ties by
minimum SHA256 of canonical JSON of sorted task IDs, then the sorted ID tuple.
Neither baseline accuracy nor subjective prompt review nor GEPA outcomes enter
selection. At most five tasks come from one shared source state; they are not
independent source-state replicates.

## Stop, audit and subsequent phase

Acquisition stops automatically after baseline initialization. A separate
ZERO-API audit verifies source freeze, exact allowlisted requests, durable
attempt/result ledger, response identities, token accounting, split isolation,
complete task integrity and deterministic reconstruction. If fewer than four
tasks qualify, retain PARENT_CATALOG_INSUFFICIENT and HOLD, with no extra APIs.
Otherwise freeze exactly four task identities and generate new Phase-B draft
preregistration inputs. Phase B is not authorized and no 4x8 runner is added.

Public artifacts contain only IDs, hashes, counts, decisions and costs. Complete
tasks and raw observations stay in ignored private runtime storage. Parent
acquisition cost is separate from Layer-1 optimization cost and contributes zero
to the future acceptance denominator (accepted mutations / contract-valid
changed proposals reaching Solver).

The execution handoff additionally freezes the clean implementation commit,
source hashes, this protocol, manifest, private input hash and exact command.
The API authorization is limited to Phase A by the user's explicit instruction.
