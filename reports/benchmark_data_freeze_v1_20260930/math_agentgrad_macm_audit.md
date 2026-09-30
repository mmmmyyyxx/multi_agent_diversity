# AgentGrad / MACM MATH audit

Status: **AGENTGRAD_MATH_EXACT_SPLIT_UNRESOLVED**.

AgentGrad v2 section 5.1 states that four benchmarks adopt GEPA's systems,
splits and rewards, while MATH adopts MACM. It supplies no recoverable MATH
sample IDs, source split, sampling seed, counts for optimizer/gate/test groups,
or equivalence evaluator configuration. The downloaded TeX source archive was
inspected: its appendix inputs are commented out and their files are absent.
No author code/config/experiment-output artifact linked from the inspected
paper or source supplies these identities. This is a search outcome, not a
claim that no unreleased artifact can exist. [AgentGrad v2, §5.1](https://arxiv.org/html/2609.08572v2).

MACM Appendix B describes random one-third selection for its general MATH
evaluation, and random half of each category's level-5 questions for a separate
difficult-problem evaluation. It does not give a deterministic membership,
seed, canonical train/test origin, or optimizer/shadow/test protocol. These
ratios cannot recover AgentGrad's exact groups. [MACM, Appendix B](https://arxiv.org/html/2404.04735).

The MACM repository was inspected at
`bfb98e7742320bc5dfc8e812532f7bfdc010dffb`. `main.py:evaluate_dataset` gathers
JSON paths with `os.walk`, uses unseeded global `random.shuffle`, and has a
default limit of five. The entrypoint instead asks for a user question. There
are no retained sample-ID/config/output files in that tree. The routine prints
solutions but does not implement a frozen reference-equivalence scoring
endpoint. Its answer regex and judges are system components, not proof of a
trusted final evaluator. [Pinned MACM source](https://github.com/bin123apple/MACM/blob/bfb98e7742320bc5dfc8e812532f7bfdc010dffb/main.py).

## Canonical source and proposed project fallback

The official author repository links to `hendrycks/competition_math`. Its
resolved revision is `71b758ecc688b2822d07ffa7f8393299f1dc7cac`, but the anonymous
MATH.zip probe returned HTTP 401. The preferred qwedsacf mirror at
`e839825f9ec5c6cfa585c654a59610969ec13993` exposes one combined train parquet,
without verified original train/test membership. Neither was silently treated
as a usable canonical split. [Author repository](https://github.com/hendrycks/math).

The pending alternative is `EleutherAI/hendrycks_math` revision
`21a5633873b6a120296cce3e2df9d5550074f4a3`, with seven original subject
configurations. Materialization fixes subject-config order and preserves each
parquet's row order, then verifies 7,500 train / 5,000 test. It remains
**not downloaded**, so canonical counts, source distribution, reference answer
extraction and final split hashes are unverified. The proposed 150/300/0/300
seed-one project split is tested only on synthetic fixtures. No exact
AgentGrad/MACM identity is claimed.

## Evaluator audit

MATH evaluator status remains **UNFROZEN**. Reference extraction finds the last
balanced boxed/fbox payload; it is not a correctness or equivalence evaluator.
The original repository includes evaluation code, but no current project
configuration has frozen its behavior. MACM's published code does not resolve
this; AgentGrad does not specify enough details.

The pinned GEPA `pyproject.toml` line 53 specifies `math-verify==0.6.0`.
Its LiveBench AMPS component also contains SymPy-based parsing/equivalence with
timeouts, adapted from the community Minerva evaluator. A dependency pin or a
different benchmark's code is insufficient to freeze this project's MATH
evaluator. Math-Verify documents extraction, symbolic conversion and gold
comparison as separate stages; these settings and invalid-output behavior
require a scientific protocol decision. [Math-Verify documentation](https://github.com/huggingface/Math-Verify).
