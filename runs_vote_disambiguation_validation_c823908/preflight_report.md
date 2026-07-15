# Preflight Report

- Git commit: `c823908f20071c56884f6c2e557b9992d49df365`
- Tracked-source dirty: `false` (only untracked historical run directories were present)
- Tests: `156 passed in 1.85s`
- Active runner: none; PID 9160 is an exited Windows process object (`HasExited=true`)
- Effective protocol: `vote_oriented_v5`
- Attachment's requested `vote_oriented_v1`: historical label; not used for compatibility because current code and Stage 2 are v4
- Checkpoint version: `2`
- Vote tie-break: `random`
- Models: solver=`deepseek-chat`, prompt generator=`deepseek-chat`, critic=`deepseek-chat`
- Candidate evaluation source: `optimization_train`
- Validation role: `vote_first` state selection only
- Test role: one final evaluation after restoring selected state

## Strict split

| Split | Count | SHA256 |
|---|---:|---|
| opt | 75 | `5491dfdf103802eb02aa49521dbde52d41a3bd2acbdfcf72837260de5033880c` |
| val | 50 | `79591c8614188670757849a96c65742e34a03314cb4299568c9f917be96d2ccb` |
| test | 125 | `e42c3f3d93583aed353099f4e28c012ff62028f682ebdb758435b4c8755e3ae0` |

Overlaps: opt/val=0, opt/test=0, val/test=0.

## Reuse

Six completed Stage 1/2 run records were selected. Other-task Stage 2 runs are excluded and left untouched; see `REUSED_RUNS.json` and `EXCLUDED_RUNS.json`.
