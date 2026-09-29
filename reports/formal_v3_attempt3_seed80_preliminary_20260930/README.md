# Formal V3 attempt3: Seed80 preliminary search report

## Scope and status

Only the two preregistered Seed80 cells were executed, once each and in the
specified order: Native, then Responsibility-only Layer2. Both used execution
source `9e498496abff9228bd10697d8d97d36f56fca3f9`. This report is a
read-only preliminary analysis of those completed searches. Seed81 and Seed82
are on hold pending renewed explicit authorization. No Validation50 or Test50
evaluation was performed.

| Cell | Frozen stop | Successful provider calls | Failed attempts | Final team |
| --- | --- | ---: | ---: | --- |
| Seed80 Native | `SATURATION_REACHED` | 2,272 | 0 | Official GEPA returned candidate replicated to five members |
| Seed80 Layer2 | `NO_FEASIBLE_LAYER2_OPPORTUNITY` | 7,190 | 0 | Four atomic commits; final five-member team preserved |

The two cells started from the same initial team hash
`6a7200eb782f005306d996f0be942d1541c859ba0394a65e1ea42c4a5d6d7c89`.
Their final team hashes are different. Native's is
`870fc8b1837c53e4847df659606fe90ecc0220c3df9718769a6eb9dc9525f8d7`;
Layer2's is
`26c38986955cbe16156eed92866c5e97bdaaa8b170aacf0297441bd3057cde95`.

## Scientific validity and protocol conformance

**Seed80 Native: VALID / PASS.** The frozen run ended at saturation. Its
authorization was consumed once. Lifecycle, ledger, final-team materialization,
51-file raw inventory, and one-record derived trajectory verify against their
frozen hashes. Its protocol and run identities match the preregistration.

**Seed80 Layer2: VALID / PASS.** The frozen run ended when no member could form
the required unchanged TeamMiniBatch and evidence packet. The terminal
feasibility trace has an empty eligible-member set. Six completed opportunities
form a continuous parent-to-child team-hash chain, with four atomic commits.
Authorization, lifecycle, ledger, final-team materialization, 150-file raw
inventory, and 31-record derived trajectory verify. No emergency ceiling was
reached. This stopping outcome is a preregistered scientific stop, not an
execution failure.

Both searches made **zero Validation50 calls and zero Test50 calls**. Their
physical ledgers show no held-out evaluation phase. The separate mechanical
audits in this directory contain the identities, counters, verification checks,
and evidence hashes. Raw run files remain in ignored local `runs/` storage;
they are not published here.

## Preliminary search observations

Layer2 returned 24 local candidates across six opportunities. Twelve passed
TeamMiniBatch, five passed ordinary Common-Safe, four reached winner-only
Shadow, and those four were committed. The committed targets were members
1, 3, 4, and 0. At the terminal check, three members lacked the required
packet repair capacity and two lacked the primary repair quota. The final
no-feasible stop followed those commits.

The adaptive Shadow50 gate recorded 23/50 vote-correct for the incumbent at
the first committed transition and 28/50 for the final committed state. These
counts describe the search path only: Shadow50 is part of the adaptive
write-back gate. They are **not** Validation50 results or an independent
generalization estimate. Three committed Shadow checks were vote-neutral;
one increased the gate's vote-correct count by five.

Layer2 used 7,190 physical provider calls versus Native's 2,272, about 3.16
times as many. This is a descriptive resource count. The arms have different
native optimization units and data flows, so it is not a matched-budget
efficiency estimate.

**Efficacy: NOT_EVALUABLE at this stage.** The available evidence is one seed
and adaptive-search observations. Neither final team has a separately
authorized paired Validation50 evaluation; there is no basis yet for a
held-out VoteAcc comparison or a superiority claim. No run was repeated or
adapted in response to the observed outcomes.

## Frozen evidence references

| Cell | Raw inventory SHA-256 | Derived trajectory SHA-256 |
| --- | --- | --- |
| Native | `3fad3e6344d8c7b256b4227eaf8d35489ea5ebb7e3183cf00a208c0f43785c4d` | `fde9bcdd40bb701c18570af3fc3a030e376733504b42e7442ee1ba95b5083c1c` |
| Layer2 | `72deab85345d1677c9fbb221bad571c0423e4d6d7507ebad0db42a46bd8efead` | `742b76e61fad2b19d392e13363172b5307afd362acd71fd0e58c3d77db098674` |

The raw inventories include execution summaries, consumed authorizations,
lifecycles, physical ledgers, final-team materializations, and supporting
traces. The published JSON audits intentionally omit prompts, examples,
answers, model outputs, credentials, endpoints, and machine-local paths.
