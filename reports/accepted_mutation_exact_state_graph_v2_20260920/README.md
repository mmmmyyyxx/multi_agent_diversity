# Accepted-mutation exact state graph v2

Cache-only recovery stopped with `CACHE_ONLY_PROFILE_RECOVERY_INCOMPLETE`. All 500 required exact requests are present as successful historical ledger events, but their responses survived only in the v2 process-local cache. No provider call was made and no exact graph was fabricated. A future request replay would be a prospective realization, not recovery of the historical outputs. The bounded v1 proof remains authoritative.
