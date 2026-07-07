# Best Prompt Accuracy

Accuracy is computed on each task final test predictions from `shared_guarded_beam_seed42`. Unique-prompt rows group agents with the same prompt hash.

## boolean_expressions

| prompt_hash | agents | accuracy | correct/total | prompt summary |
|---|---:|---:|---:|---|
| 00bcb3ee9efb | 0,2,3 | 0.9800 | 294/300 | base careful reasoning solver |
| b22fd4028486 | 1 | 0.9800 | 98/100 | base solver + distinct decision procedure repeated x1 |
| 4e7f4fa02b07 | 4 | 0.9600 | 96/100 | base solver + distinct decision procedure repeated x3 |

## disambiguation_qa

| prompt_hash | agents | accuracy | correct/total | prompt summary |
|---|---:|---:|---:|---|
| 3ed6e151874c | 1,3 | 0.6900 | 138/200 | base solver + distinct decision procedure repeated x10 |
| 9aa56bc033b1 | 0 | 0.6900 | 69/100 | base solver + distinct decision procedure repeated x11 |
| b22fd4028486 | 2 | 0.5100 | 51/100 | base solver + distinct decision procedure repeated x1 |
| 00bcb3ee9efb | 4 | 0.4400 | 44/100 | base careful reasoning solver |

## formal_fallacies

| prompt_hash | agents | accuracy | correct/total | prompt summary |
|---|---:|---:|---:|---|
| dce5e63f71f6 | 3 | 1.0000 | 100/100 | base solver + distinct decision procedure repeated x2 |
| b22fd4028486 | 0,1 | 0.9950 | 199/200 | base solver + distinct decision procedure repeated x1 |
| 4e7f4fa02b07 | 2 | 0.9800 | 98/100 | base solver + distinct decision procedure repeated x3 |
| 5ba21276572f | 4 | 0.9800 | 98/100 | base solver + distinct decision procedure repeated x6 |

## geometric_shapes

| prompt_hash | agents | accuracy | correct/total | prompt summary |
|---|---:|---:|---:|---|
| b04748fb68db | 0 | 0.7600 | 76/100 | geometric primitive counter; expand SVG commands, count primitives, classify by closure/vertices |
| dce5e63f71f6 | 3,4 | 0.7250 | 145/200 | base solver + distinct decision procedure repeated x2 |
| ffbdf3d66222 | 2 | 0.7200 | 72/100 | base solver + distinct decision procedure repeated x4 |
| 00bcb3ee9efb | 1 | 0.6600 | 66/100 | base careful reasoning solver |

## ruin_names

| prompt_hash | agents | accuracy | correct/total | prompt summary |
|---|---:|---:|---:|---|
| 5ba21276572f | 2 | 0.8200 | 82/100 | base solver + distinct decision procedure repeated x6 |
| 7aa15b6bcde7 | 4 | 0.8200 | 82/100 | base solver + distinct decision procedure repeated x12 |
| b24374b21beb | 3 | 0.8100 | 81/100 | base solver + distinct decision procedure repeated x5 |
| 00bcb3ee9efb | 0,1 | 0.7900 | 158/200 | base careful reasoning solver |

## sports_understanding

| prompt_hash | agents | accuracy | correct/total | prompt summary |
|---|---:|---:|---:|---|
| 00bcb3ee9efb | 3,4 | 0.8300 | 166/200 | base careful reasoning solver |
| dce5e63f71f6 | 2 | 0.8300 | 83/100 | base solver + distinct decision procedure repeated x2 |
| b24374b21beb | 1 | 0.8200 | 82/100 | base solver + distinct decision procedure repeated x5 |
| b22fd4028486 | 0 | 0.8100 | 81/100 | base solver + distinct decision procedure repeated x1 |
