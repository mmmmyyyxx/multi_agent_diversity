# Responsibility compatibility

The current D/N/C diagnostic needs boolean member and team correctness, an equal-weight discrete plurality state, counterfactual member vote effects, meaningful near-margin and coverage, and the existing 4/4/4 evidence construction. None of the five project protocols has all of these frozen and implemented. HotpotQA answer EM yields a provisional boolean member score and a normalized vote class, but task choice, real split and state/evidence composition are absent; its registry explicitly sets `supports_current_responsibility=False`. HoVer, IFBench, PUPA and MATH are likewise false.

`UnifiedSearchOrchestrator.run` checks registered benchmark blockers before snapshot, diagnosis, engine, provider, or gate access. The previous LLM-aggregation-versus-plurality-responsibility guard remains. The five fake tests supply components that throw if touched and confirm `HOLD_PRE_PROVIDER` instead.
