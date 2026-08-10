# Module 3 Fully Offline Diagnosis

## 1. Executive conclusion

Seed46 shows a mixed S2→S3 degradation rather than a Layer-3-threshold failure.
Both exact replays pass with zero constraint, branch-winner, or global-commit
mismatch. On the fixed S3 Stage-B pool, common and RCRU feasibility differ for
5 of 88 candidates,
and the hypothetical global winner changes on 4 of 32 updates.
RCRU positive support crosses the true plurality boundary in
10/44 cases, while 34/44
cases are coverage or margin progress that remains pre-boundary. S3 still raises
train vote 50-to-58, but S2 reaches 50-to-60 and has more accepted transitions.
The primary evidence therefore points to `MIXED` across
feasibility/ranking differences and boundary-credit alignment. Generation
attrition is recorded only as a secondary, trajectory-confounded signal. This
does not authorize a v15 change or prove multi-seed harm.

## 2. Agent1/4 zero-commit diagnosis

| Agent | Selected | Candidate-producing branches | Stage-B candidates | L1 | L2 | L3 | Branch winners | Competition losses | Commits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 1 | 2 | 2 | 2 | 2 | 1 | 0 | 1 |
| 1 | 17 | 16 | 31 | 1 | 1 | 1 | 1 | 1 | 0 |
| 2 | 18 | 9 | 17 | 8 | 5 | 5 | 4 | 1 | 3 |
| 3 | 14 | 7 | 12 | 8 | 6 | 6 | 3 | 0 | 3 |
| 4 | 14 | 13 | 26 | 0 | 0 | 0 | 0 | 0 | 0 |


Agent1 event outcomes: `{"COMMON_SAFETY_FAILURE": 15, "CROSS_BRANCH_COMPETITION_LOSS": 1, "GENERATION_FAILURE": 1}`. Its primary
classification is **COMMON_SAFETY**.

Agent4 event outcomes: `{"COMMON_SAFETY_FAILURE": 13, "GENERATION_FAILURE": 1}`. Its primary
classification is **COMMON_SAFETY**.

The repeated-target timeline shows the exact pre-selection failure discount,
expected update value, active lane, candidate yield, and terminal stage in
`repeated_target_failure_timeline.csv`. This separates persistent selector
opportunity from realized generator/RCRU feasibility.

Agent1 reached failure count 8 and discount
0.111, yet remained rank 1 on
12 selections and rank 2 on
5; its lane mix was
`{"coverage": 10, "direct_flip": 6, "margin_support": 1}`. Agent4 reached
failure count 9 and discount
0.100, while remaining rank 1 on
1 selection and rank 2 on
13; its lane mix was
`{"coverage": 3, "direct_flip": 1, "margin_support": 10}`. The normalized
opportunity ranking therefore kept both agents actionable despite discount,
but 28/31 of their selected branches failed the common target/vote safety
policy. Only 2/31 were generation failures, and neither agent had a
common-feasible candidate rejected only by the RCRU lane policy.

## 3. Common-policy counterfactual

Both replay validations are **PASS**. Candidate-set counts are:

- both feasible: 11
- common-only: 2
- RCRU-only: 3
- neither: 72
- common-only target-gain/no-vote/no-lane subtype: 2
- RCRU-only lane-only subtype: 3

The one-step hypothetical global winner changes on 4/32 updates.
This result freezes each actual S3 parent, target pair, generated candidate pool,
and completed rollout, then changes only the decision policy. It is explicitly
**not** a chained S2 trajectory and cannot imply test=93.

Important changed candidate hashes are listed in
`counterfactual_commit_mismatches.csv`; the complete 88-candidate decision table
is `candidate_policy_counterfactual.jsonl`.

The four changed updates include two common-only target improvements (updates 1
and 22), one RCRU-only lane commit (update 8), and one same-objective branch
ranking change (update 4). The fixed-pool regret summary records one update
where common policy would commit while RCRU commits nothing, one where the
common winner has higher vote gain, and two where equal-vote common winners
have higher target gain.

## 4. Boundary alignment diagnosis

The true boundary is `M0 <= 0 -> M1 > 0`, not merely `G 2->3`.
Positive-support boundary-cross fraction is
10/44 = 0.227;
pre-boundary coverage/margin fraction is
34/44 = 0.773.
Margin saturation is **not observed**
with 0 real `(M0, utility_delta)`
witness groups containing both `M1=0` and `M1>0` outcomes.

## 5. S2 vs S3 transition structure

| Setting | Accepted | Vote gains | Vote losses | Net train vote | Boundary crosses | Member-only | Lane-only |
|---|---:|---:|---:|---:|---:|---:|---:|
| S2 | 9 | 15 | 5 | +10 | 15 | 4 | 0 |
| S3 | 7 | 8 | 0 | +8 | 8 | 3 | 1 |

The final test association is S2=93 and S3=87, but test observations never enter
the replay or any ranking decision.

## 6. Hypothesis verdicts

- **H1: NOT_SUPPORTED** — `{"evidence": {"1": {"COMMON_SAFETY_FAILURE": 15, "CROSS_BRANCH_COMPETITION_LOSS": 1, "GENERATION_FAILURE": 1}, "4": {"COMMON_SAFETY_FAILURE": 13, "GENERATION_FAILURE": 1}}, "interpretation": "Generation failure occurred in only 2/31 Agent1/4 selections; 28/31 ended in common target/vote safety failure.", "verdict": "NOT_SUPPORTED"}`
- **H2: NOT_SUPPORTED** — `{"common_only_candidates_agent1_agent4": 0, "verdict": "NOT_SUPPORTED"}`
- **H3: SUPPORTED** — `{"rcru_lane_only_commit_count": 1, "rcru_only_lane_only_candidate_count": 3, "verdict": "SUPPORTED"}`
- **H4: SUPPORTED** — `{"boundary_cross_count": 10, "positive_support_total": 44, "preboundary_count": 34, "verdict": "SUPPORTED"}`
- **H5: NOT_SUPPORTED** — `{"saturation_witness_group_count": 0, "verdict": "NOT_SUPPORTED"}`
- **H6: PARTIALLY_SUPPORTED** — `{"branch_winner_lane_positive_coalition_zero": 5, "commit_lane_positive_coalition_zero": 4, "limitation": "Zero coalition delta shows ranking tolerance, not causal dilution by itself.", "verdict": "PARTIALLY_SUPPORTED"}`
- **H7: SUPPORTED** — `{"layer2_pass": 14, "layer3_pass": 14, "verdict": "SUPPORTED"}`


## 7. What the evidence does not prove

- One seed cannot establish that RCRU or Module 3 is generally harmful.
- A fixed-parent one-step counterfactual is not an alternative training trajectory.
- Candidate generation differs between actual S2 and S3, so this analysis only
  isolates decision-policy effects within the observed S3 Stage-B pool.
- Correlations between ranking signals and boundary conversion are descriptive,
  not causal.
- No v15 design change is authorized by this diagnosis.

## Machine-readable conclusion

See `module3_diagnosis_summary.json`. `API_CALLS = 0`.
