# Gate 1 Public Summary

- Source commit: `ad3ba73a89cc579bae906c570a677c72600d955c`
- Local source artifact: `runs_gate1_invalid_recovery_ad3ba73_20260725_124426`
- Cases: 14
- Repetitions: 3 per case
- Records: 42
- Eventual valid rate: `1.0`
- Terminal invalid count: `0`
- Recovered requests: `1`
- Recovery API calls: `1`
- Recovery tokens: `633`
- Recovery call overhead: `0.0238095238`
- Recovery token overhead: `0.0246639392`
- Cache replay created new Solver calls: `false`
- Recovery summaries reconciled: `true`

The one recovered case was `historical_missing_final_04`. Its first attempt
had `finish_reason=length` and `missing_final_answer`; the request-local
recovery was valid. Raw responses, SQLite caches, and runner logs remain local.
