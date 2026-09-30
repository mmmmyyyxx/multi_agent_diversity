# Split exactness

IFBench is `EXACT_SOURCE_AND_INSTANCE_ALIGNED`: exact pinned vendored files,
shadow first 300 train rows, search independent seed-1 sample of rows 300:600,
entire 294-row test file. Its source byte hashes and Git blob IDs are retained.

HotpotQA and HoVer are pending download. The reproducible recipes preserve
source order, use the GEPA .4/.8 pools and independent seed-1 trims; HoVer first
filters unique supporting-document count to three and shuffles with seed zero.
After successful actual materialization their defensible label is
`SOURCE_LOGIC_REPRODUCED`. The original paper's HF revisions and fingerprints
have not been recovered. A newly pinned revision is not historical instance
identity.

PUPA is also pending. It uses the inspected sequential pupa_new 111/111/221
slices, plus a separately retained auxiliary pupa_tnb source. The GEPA source
does not pin its original HF revision. Therefore current HF revision and the
same slice algorithm alone support `SOURCE_LOGIC_REPRODUCED`; an exact original
paper byte claim would require additional evidence.

MATH is pending. Its `PROJECT_MATH_V1_PROPOSED` recipe independently shuffles
canonical train/test row indexes with seed one, then takes train[:150],
train[150:450] and test[:300]. It is `PROJECT_PROPOSED`, not an AgentGrad or MACM
replication. See the MATH audit. All V1 recipes leave independent validation
empty; adaptive shadow is not an independent final validation endpoint.
