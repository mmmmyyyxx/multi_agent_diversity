# Sequential Symmetry-Breaking Online Pilot v1 Handoff

`READY_TO_RUN=false`

The scientific protocol, runner semantics, fixed parent source, endpoint
definitions, stop rules, split access, and hard ceilings are frozen. Real API
execution remains blocked because `api_authorization.authorized=false` and the
manifest lifecycle has not entered `RUNNING`.

After explicit authorization for this exact pilot, Sol must make an
authorization-only manifest/handoff refreeze, set the lifecycle to `RUNNING`,
record the exact implementation commit and preregistration hash, generate a
fresh zero-API prep root, and independently verify the source and private-input
hashes. The execution handoff may then contain only this command:

```powershell
$env:SEQUENTIAL_SYMMETRY_BREAKING_V1_AUTHORIZED='1'
python scripts\run_sequential_symmetry_breaking_online_pilot_v1.py --execute --prep runs/sequential_symmetry_breaking_online_pilot_v1_prep_authorized --run runs/sequential_symmetry_breaking_online_pilot_v1_attempt1
```

The execution agent must use the exact frozen command, make no tracked edits,
and stop on any mismatch or failure. Validation50 and Test50 access are zero.
There is no automatic retry, resume, budget extension, or experiment rerun.
