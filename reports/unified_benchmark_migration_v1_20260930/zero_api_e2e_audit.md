# Zero API path audit

The credential-free `tests/formal_zero_api_runner.py` installs the socket guard before application import, checks its negative control, and executes the focused fake suite. HotpotQA fixture exercises parse, normalize, EM/F1, five-member plurality, fake LLM aggregate and independent aggregator accounting. Each of the five requested benchmark adapters exercises the orchestrator's pre-provider hold. The four unselected adapters also refuse format/parse/score operations. The production CLI preflight lists benchmark-specific blockers and zero provider attempts.

No real provider, reflection, aggregator or Validation50/Test50 request was made. The fake LLM request contains only public problem, public context, output contract and equal-status responses. The gold string is absent from the request payload. Cache keys differ by benchmark, version, parser contract, role, seed and output contract.

`30/30` focused tests and `1661/1661` runnable broad tests passed; four broad tests skipped. The guard reported zero network attempts. Fake LLM aggregation accounts for one logical evaluation and one physical call, or one cache hit and zero physical calls. Solver and reflection counts remain separate in the existing search result contract; no new benchmark search path consumed either role.
