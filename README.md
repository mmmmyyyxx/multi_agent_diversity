# Multi-agent Prompt Search

Jointly optimize a prompt team with frozen model weights and benchmark-defined
evaluation. The sole active research architecture is Unified Team Prompt Search.
The scientific specification is [CURRENT_SPEC](docs/design/CURRENT_SPEC.md).

```mermaid
flowchart LR
  B[Benchmark adapter] --> S[Team snapshot]
  S --> D[Diagnosis and opportunity]
  D --> E[Replaceable search engine]
  E --> C[Candidate evaluation and aggregation]
  C --> T[Transition and adaptive gate]
  T --> H[Atomic commit and history]
  H --> G[Global stop]
  G --> S
```

## Start here

Read [AGENTS](AGENTS.md), [repository map](docs/REPOSITORY_MAP.md),
[method overview](method.md) and [current architecture](docs/CURRENT_ARCHITECTURE.md).
Current experiment state is in [current frontier](experiments/current_frontier.yaml)
and [research state](docs/research/CURRENT_RESEARCH_STATE.md).

## Offline quick start

```powershell
python scripts/audit_repository_governance.py --check-generated
python tests/formal_zero_api_runner.py tests --suite=current
python scripts/run_experiment.py --help
```

The sole current execution entrypoint is scripts/run_experiment.py. Real execution
requires a governed frozen prep and explicit attempt-specific authorization.
Test commands use fake providers and a pre-import network guard.

## Governance and evidence

[Registry](experiments/registry.yaml) records metadata; [lineage authority](experiments/lineage.yaml)
generates [LINEAGE](docs/experiments/LINEAGE.md). New experiments use the
[v2 template](experiments/templates/unified_experiment_v2.yaml).
[Reports index](reports/INDEX.md) catalogs immutable evidence. Reports are
evidence, never design authority. [Archives](docs/archive/) preserve historical
contracts and paper text with source provenance.

## Current V2 method state

The active opt-in method is `unified_team_prompt_search_v2`. Team admission is
independent of strict GEPA local survival; evidence sizes are variable. Pattern
and deterministic private/shared Memory are implemented, default disabled.
Use `SearchMethodConfig.v2()` and the V2 composition; the V1 constructor and
adapters remain reproduction paths. See the normative CURRENT_SPEC for policy
boundaries. Mechanism correctness is tested offline; efficacy is unverified.
Benchmark blockers remain unchanged; no new real execution is ready or authorized.
