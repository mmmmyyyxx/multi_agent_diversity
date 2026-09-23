# Before: unified execution-governance binding

Base: `f762545e3e4c49a7ba9f8cac53b380f14c606332`.

`scripts/run_experiment.py` parses `scientific` and `runtime`, checks their
typed shape, and returns `HOLD`. It does not replay a disk freeze, source
files, GEPA dependency, authorization, provider fingerprint, model binding,
phase/roles, or one-time use. `--execute` unconditionally aborts before the
provider boundary. A manifest-supplied `execution_factory` is informational
only and cannot execute.

The canonical round-trip and authorization checks currently live in
`governance/startup_identity.py`, used by the historical GEPA canary runner.
That runner checks source, protocol, split and startup identity before its
run-local `RUNNING` transition. The unified entry does not yet call this
governance path. No provider can currently be constructed through the unified
entry, and it cannot enter `RUNNING`; this is safe but not execution-ready.

`provider_factory.py` and `llm_client.py` can construct clients when invoked,
so a future composition path must be reachable only after one validated,
consumed execution permit. Passing arbitrary strings in `RuntimeContext` is
not sufficient authorization.
