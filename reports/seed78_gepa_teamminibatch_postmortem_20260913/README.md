# Seed78 Local-GEPA to TeamMiniBatch postmortem

This is a zero-API, read-only audit of the completed Seed78 A/B pilot. It does
not modify the scheduler, GEPA, promotion gate, frozen run, or historical report.

## Finding

`ROOT_CANDIDATE_RETURN_SEMANTICS_CONFIRMED`

Across 24 target branches, GEPA emitted 89 proposals and rejected all 89 under
its frozen strict local-improvement rule. Every persisted GEPA state contained
only program candidate index 0; every per-instance frontier pointed to index 0;
and every candidate returned to Layer 2 was byte-identical to its parent.

Parent-versus-parent TeamMiniBatch evaluation has exact zero deltas for invalid,
Vote, Target, Coalition, Responsibility, and Broad signals. Therefore all 24
promotion failures are classified `UNCHANGED_PARENT`; none supplies evidence
that the TeamMiniBatch thresholds were too strict or that a changed local
mutation failed team transfer.

The scheduler changed allocation, but its causal efficacy remains not evaluated
because no changed candidate reached a write-back decision.

## Paired final evaluation

The two final teams and their validation request-identity sets were identical.
The historical runner nevertheless used independent per-arm caches and made 50
successful validation provider calls per arm. Its 0.62 versus 0.60 result is
therefore not a scheduler contrast. Future paired experiments must share one
realization cache for identical prompt/question/contract identities.

## Boundaries

API calls = 0. Validation calls = 0. Test calls = 0. No raw prompt, question,
answer, response, endpoint, credential, cache, checkpoint, or absolute path is
published here.
