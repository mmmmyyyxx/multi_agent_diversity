# Seed81 V4 attempt3 execution and scientific audit

## Frozen identity and raw evidence

The single authorized attempt was
`gepa_layer2_local_to_team_transfer_diagnostic_v4_seed81_attempt3` at execution
source `66e1762ad7ea8d85f71de3b604d0e04fb8b9c7d0`, protocol SHA-256
`a9a0d72715029ac78e3ab46533f57faa2c0adf2f699c862becc91027de35c20f`.
Seed 81 used the frozen Responsibility-only Layer2 V4 method, `qwen3-8b`
Solver with thinking disabled, `qwen3.7-flash` Reflection, and the `lwj`
provider profile. Validation50 and Test50 had zero calls.

The run exited with lifecycle `EXECUTION_COMPLETE`. Its raw inventory was
written immediately after exit and covers 83 files: the consumed authorization,
both prep copies, raw console output, ledger, lifecycle, execution summary,
GEPA states, lineage and trajectory evidence. Inventory SHA-256:
`a907c155c720f2d218f393e46594bac7261595a54e7953d3628e01c4a47e979b`.
All 83 file sizes and SHA-256 values were independently replayed with zero
mismatches. Raw files remain under the ignored attempt3 `runs/` root and are
not copied into this report.

| Frozen raw artifact | SHA-256 |
| --- | --- |
| Ledger | `cb55b86adb1b8b5c687762ca3c7f6f5c6a722a758e7bce32981bf8499a1c3d0d` |
| Lifecycle | `59159ca3bfa29c211c7b4a54c501fdd625e4515044af4ab4836761775612f67f` |
| Execution summary | `296aafc4dc390ff3e409b3f5fe9a66c022926644dee8832e87fe4346fddaeccf` |
| GEPA lineage, opportunities 1/2/3 | `8fe7c2cc5420cbbf8b3af3c9834a2a81ae936d8f063e1d15fcfa65caae117b91` / `1fc20bc63f72d50e8f2cd6ed28eb2b930607eeb9b54e976d89f56d34bbce6fee` / `0cda0746522438cfc479491a8e4e932a91049d0e9ef475e54de8ab1a61c7cfaf` |

## Protocol conformance

The independent [machine-readable audit](scientific_validity_audit.json)
replayed source and protocol identity, one-time authorization consumption,
split hashes, frozen GEPA dependency, raw artifact hashes, ledger and lifecycle
accounting, GEPA lineage/state/frontier, returned-candidate identity, mandatory
Full, and ordinary admission. It returned `PASS`.

Three complete opportunities produced 2, 4, and 4 reflection proposals.
The total was 10. The preregistered pre-opportunity guard reserves up to 12
proposals for the next complete GEPA search, and `10 + 12 > 20`; it therefore
stopped before a fourth opportunity. This is an authorized scientific stop,
not an execution failure. There was no candidate truncation within an
opportunity. The actual sample was one
`LAYER1_RETURNED_STRICT_POSITIVE_CANDIDATE`, below the prospective target of
five. Internal GEPA accepted mutations also totaled one; the counts were
audited separately and were not pooled with attempt2.

The durable ledger recorded 213 provider attempts, all successful, 493 cache
hits, and no failed attempt or postprocess failure. The emergency ceilings
were 7,000 successful calls and 150,000 attempts. Ledger phases contained no
Validation50 or Test50 access. The 337 calls from invalid attempt2 were
excluded entirely.

## Observation and classification

The sole returned candidate had strict-positive local acceptance delta `+2`.
Ordinary TeamMiniBatch promoted it. Ordinary Full found target-member correct
count delta `+10` and team vote-correct delta `0`; Common-Safe failed. Shadow
was not reached and no prompt was committed. All three opportunities retained
the same parent team hash. There was no diagnostic-only Full in this realized
sample because the only candidate was ordinarily promoted.

**Scientific validity: `VALID`.** The run followed the preregistered early
stop and all audited execution, evidence, access, and admission rules.

**Efficacy: `NOT_EVALUABLE`.** Only one of the prospective five returned
candidates was observed. The one candidate is descriptively local-positive
and team-vote-neutral, but it cannot establish the planned transfer result or
a general conclusion about V4 efficacy.

This authorization is consumed. No rerun is justified by the observed effect
or the allowed early stop. Formal V3 remains blocked, and no additional real
API work was started.
