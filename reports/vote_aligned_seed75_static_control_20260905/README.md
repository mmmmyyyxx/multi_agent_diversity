# Seed75 Static No-Training Control

The Static no-training control was added after observing the preregistered
Seed75 P0/P1 pilot. It only decomposes absolute change from the exact initial
ensemble and relative change from Generic optimization. It does not alter the
original `NO_CLEAR_SIGNAL` classification.

Static exactly matched the P0/P1 frozen initialization. Optimize100 reused the
existing frozen rollout with zero new provider calls. Shadow50 and Validation50
were evaluated once each. Static performed zero updates, target selections,
candidate generation, revisions, write-back decisions, or commits. Test50 was
not accessed.

## Validation50

| Arm | Vote | MeanMember | Vote-MeanMember | Oracle |
|---|---:|---:|---:|---:|
| Static | 0.560 | 0.560 | 0.000 | 0.560 |
| P0 | 0.640 | 0.652 | -0.012 | 0.980 |
| P1 | 0.700 | 0.636 | 0.064 | 0.800 |

P1 minus Static Validation deltas are Vote `+0.140`,
MeanMember `+0.076`, Vote-minus-MeanMember
`+0.064`, and Oracle
`+0.240`.

Supplementary classifiers: `P1_ABSOLUTE_MEMBER_GAIN` and ensemble-structure
`POSITIVE` (`CASE_1_INDIVIDUAL_AND_ENSEMBLE_GAIN`). These are descriptive
single-seed labels, not a revision of the original pilot decision.

The key ambiguity is therefore resolved in favor of relative
under-improvement: P1 does not reduce average member competence below the
initial team. It improves MeanMember by 0.076, while P0 improves it by 0.092.
P1 gives up 0.016 of P0's individual gain but adds 0.076 of ensemble structure
relative to P0. Relative to Static, P1 adds both individual competence and
useful plurality structure.

## Optimize100, Shadow50, and Validation50

| Split | Arm | Vote | MeanMember | Vote-MeanMember | Oracle |
|---|---|---:|---:|---:|---:|
| Optimize | Static | 0.590 | 0.590 | 0.000 | 0.590 |
| Optimize | P0 | 0.790 | 0.728 | 0.062 | 0.980 |
| Optimize | P1 | 0.760 | 0.712 | 0.048 | 0.980 |
| Shadow | Static | 0.400 | 0.400 | 0.000 | 0.400 |
| Shadow | P0 | 0.660 | 0.636 | 0.024 | 0.880 |
| Shadow | P1 | 0.600 | 0.600 | 0.000 | 0.860 |
| Validation | Static | 0.560 | 0.560 | 0.000 | 0.560 |
| Validation | P0 | 0.640 | 0.652 | -0.012 | 0.980 |
| Validation | P1 | 0.700 | 0.636 | 0.064 | 0.800 |

## Validation member accuracy

| Member | Static | P0 | P1 |
|---:|---:|---:|---:|
| 0 | 0.560 | 0.680 | 0.620 |
| 1 | 0.560 | 0.620 | 0.600 |
| 2 | 0.560 | 0.740 | 0.580 |
| 3 | 0.560 | 0.500 | 0.680 |
| 4 | 0.560 | 0.720 | 0.700 |

Four of five P1 members improve over Static; member 2 improves slightly. P0
has a higher mean but lowers member 3 below Static, illustrating that its broad
coverage gain is heterogeneous rather than uniform.

## Validation support depth

| Arm | G0 | G1 | G2 | G3 | G4 | G5 |
|---|---:|---:|---:|---:|---:|---:|
| Static | 22 | 0 | 0 | 0 | 0 | 28 |
| P0 | 1 | 14 | 3 | 3 | 11 | 18 |
| P1 | 10 | 3 | 2 | 8 | 7 | 20 |

The identical Static prompts produce only G0/G5 states. Both optimized arms
create differentiated support. P0 maximizes breadth (Oracle 0.98), whereas P1
retains less breadth (Oracle 0.80) but converts its differentiated support into
a higher plurality Vote. P1 Oracle is still 0.24 above Static, so P1 does not
destroy initial coverage in absolute terms.

## Isolation and calls

- Solver: `qwen3-8b`, thinking disabled; role/evaluator configuration:
  `qwen3.7-flash`.
- New Static solver provider calls: `100`.
- Teacher/Critic/Student calls: `0/0/0`.
- Additional P0/P1 training calls: `0/0`.
- New Test50 calls: `0`.

Official supplementary audit: `PASS` with 65 protected historical artifacts
unchanged. The original P0/P1 classifier remains `NO_CLEAR_SIGNAL`.
