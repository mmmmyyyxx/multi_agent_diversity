# Pre-authorization startup identity audit

Gate: **PASS**. The historical `authorized1` attempt remains an immutable `FAILED_START_AUTHORIZATION_IDENTITY_MISMATCH` with provider boundary false.

Root cause: the tracked manifest retained an earlier preregistration SHA while authorization metadata was changed; prepare hashed the changed object and runtime compared it with the stale embedded value. Authorization was also incorrectly part of the preregistration payload.

The replacement uses one canonical builder/validator, separates scientific identity from operational authorization/lifecycle metadata, validates before `RUNNING`, and has passed three-repeat, disk, fresh-process, cross-working-directory, environment, poison, stale-authorization, and post-freeze-mutation tests.

Fresh attempt: `gepa_layer2_real_canary_v2_authorized2`. State: **PREREGISTERED_NOT_EXECUTED / READY_FOR_AUTHORIZATION / AUTHORIZATION_REQUIRED**. Real API, Validation50, and Test50 calls are all zero.
