# Current Implementation Specification

The sole active research architecture is Unified Team Prompt Search.
This document is the normative active scientific method specification.
Runtime identities come from `multi_dataset_diverse_rl/versions.py`; benchmark
contracts come from `multi_dataset_diverse_rl/benchmarks/`; experiment access
and authorization come from the frozen manifest. Reports are evidence, never
design authority. Historical specifications and invariant IDs are preserved in
`docs/archive/specs/historical_current_spec.md` and its invariant index.

## Active Unified Team Prompt Search (opt-in)

- **INV-UNIFIED-FLOW-001**: One orchestrator owns snapshot, state analysis,
  opportunity construction, candidate search, progressive team evaluation,
  transition selection, adaptive validation, atomic commit, history update and
  global stopping. New method identity is a composition of versioned component
  identities, not a `native/layer2` mode.
- **INV-SEARCH-TRANSITION-001**: A search engine may use the supplied immutable
  opportunity and context to explore candidates. Team transition policy alone
  selects a candidate for write-back. This is a modularity boundary, not a
  restriction that team evidence can never enter future search research.
- **INV-GEPA-DERIVED-001**: The current derived GEPA engine preserves strict
  local improvement for migration and records it as a policy identity. The new
  architecture permits future versioned GEPA changes. Historical official
  baseline and Formal V3 frozen source retain their own exact contracts.
- **INV-EVIDENCE-ROLE-001**: Mutation, search validation, TeamProbe, Full and
  adaptive-gate evidence have distinct role-bearing views or scopes. Current BBH compatibility
  may use the same Optimize IDs for search validation and TeamProbe. Validation
  and Test never enter adaptive search without separate frozen authorization.
- **INV-RESP-CAPABILITY-001**: Current raw-overlap plurality responsibility,
  feasibility masking, raw `V=max(4D,2N,C)` and target score `V/(1+f)` remain unchanged
  during migration. A benchmark without current-responsibility capability
  fails closed; no plurality diagnostic is fabricated for LLM aggregation.
- **INV-AGGREGATION-001**: Plurality uses one equal vote per valid parsed member
  answer and abstains on top-count ties. The benchmark adapter owns parsing and
  scoring. LLM aggregation uses the optimizer model identity with a separate
  aggregator role, stage, cache key and accounting. Its request contains only
  public input, member outputs, public context and the immutable instruction;
  gold and evaluation results are not part of that interface. Its output is
  parsed again by the benchmark adapter before scoring.
- **INV-HISTORY-MEMORY-001**: Target/failure/commit counts, latest transition
  and lineage are structured optimizer history. Pattern and LLM memory ports
  default to null behavior. No held-out evaluation updates either.
- **INV-REPLAY-001**: The previous two-layer implementation remains available
  for historical reproduction and deterministic migration comparison. It is
  not the new method's ownership graph.

## Active search and stopping contract

The current migration preserves its versioned strict local acceptance,
responsibility/feasibility policies, immutable evidence roles, progressive
TeamProbe/Full evaluation, Common-Safe transition, winner-only adaptive gate,
at most one atomic prompt commit, persistent realizability and parent-scoped
team epochs. The frozen component identities in `versions.py` and the manifest
select these policies; architecture consolidation does not modify them.
Only a successful atomic team commit resets team no-update patience. A
Vote-neutral safe commit still counts. Incomplete or emergency-aborted
opportunities do not advance scientific saturation. Operational ceilings
abort execution and must never be reported as convergence.

## Benchmark interface invariants

- **INV-BENCHMARK-CONTRACT-001**: Each adapter declares its public input,
  immutable output contract, parser, scorer, capabilities and protocol identity.
  Missing required capability fails closed before provider execution.
- **INV-BENCHMARK-FIREWALL-001**: Gold, labels, scorer state and held-out data
  are inaccessible to public solver and aggregator requests. Privacy pipelines
  retain their declared trust boundaries; exceptions fail closed.
- **INV-BENCHMARK-EVALUATOR-001**: External evaluator versions/resources and
  their provenance are frozen dependencies. No download or silent alternate
  scorer is permitted in evaluation. Benchmark protocol implementations and
  IDs are authoritative in the benchmark package and `versions.py`, rather
  than duplicated here.

## Authority map

See AGENTS.md for the complete authority hierarchy; method.md is a conceptual
overview, CURRENT_ARCHITECTURE.md a code map, and reports/ immutable evidence.
Current implementation does not imply real-execution readiness or authorization.
