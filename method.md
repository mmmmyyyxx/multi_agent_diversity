# Unified Team Prompt Search

The sole active research architecture is Unified Team Prompt Search.
This is a conceptual overview. [CURRENT_SPEC](docs/design/CURRENT_SPEC.md)
defines the scientific contract and versions.py defines runtime identities.

A benchmark adapter defines public problems, parsing, scoring and capabilities.
The current five-member prompt team and its evaluated predictions become an
immutable team-state snapshot. Diagnosis computes responsibility and feasibility;
the opportunity builder selects a member and freezes role-bearing evidence.

The replaceable SearchEngine explores prompts within that opportunity. Current
derived GEPA retains its versioned strict local improvement policy. Candidate
exploration and deployable team transition are separate decisions.

The candidate pipeline evaluates a proposed single-member replacement with fixed
peers using the benchmark-selected aggregation, TeamProbe and Full scopes.
Transition policy applies its frozen safety/ranking rules; only its selected
winner enters the adaptive gate. An accepted winner commits atomically.

Structured history records failures, commits, latest transitions and epochs.
Pattern and LLM memory ports default to null. Global stopping uses the frozen
team-epoch rule; emergency ceilings are operational failure conditions.
Post-freeze Validation and sealed Test do not feed back into optimization.

Historical paper text is preserved in
[v15 method archive](docs/archive/methods/v15_method.md); historical contracts
do not define the active method. See [open questions](docs/research/OPEN_QUESTIONS.md)
for changes requiring separate scientific identities and experiments.
