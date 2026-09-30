# Before architecture

The historical production graph entered `ExperimentEngine`, selected a native
or Layer2 scope, and called a Layer2 controller with an independently owned
local optimizer. The controller assigned responsibility, built the GEPA packet,
evaluated team candidates, applied Common-Safe and Shadow, and committed one
member prompt. The local optimizer owned GEPA proposal and local acceptance.

This representation crossed its ownership boundary at evidence scheduling,
local validation, team evaluation, history, and saturation. The current
refactor was based on `BASE_SHA=a639e4402c1e314c19e5e5f782c414f4da5cf4f6`;
`origin/main=02f65a8fe22fb4473de5e9b6b65256d387a5ff2c` at start.
The tracked worktree was clean; an unrelated untracked retry report directory
was preserved.

Frozen Formal V3 entrypoints and their historical source remain available for
exact reproduction. No historical run directory or report was edited.
