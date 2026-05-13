# Sampled Test Traces (seed=42)

- Settings: shared_div, bank_div, shared_baseline, bank_baseline
- Samples per setting: 2
- Agent correctness is computed as `answers[i] == gold` from `test_epoch1_predictions.jsonl`.

## shared_div

### sample 1: `78a253d607eb`

- gold: `A`; vote: `D`; vote_correct: `0`
- answers: `['D', 'D', 'D', 'D', 'D']`

- agent 0: answer `D`, correct `False`, family `backward_reasoning` / `backward_reasoning`
  - trace excerpt: 
- agent 1: answer `D`, correct `False`, family `backward_reasoning` / `backward_reasoning`
  - trace excerpt: 
- agent 2: answer `D`, correct `False`, family `backward_reasoning` / `backward_reasoning`
  - trace excerpt: 
- agent 3: answer `D`, correct `False`, family `backward_reasoning` / `backward_reasoning`
  - trace excerpt: 
- agent 4: answer `D`, correct `False`, family `backward_reasoning` / `backward_reasoning`
  - trace excerpt: 

### sample 2: `d5ba73d6e741`

- gold: `A`; vote: `A`; vote_correct: `1`
- answers: `['A', 'A', 'A', 'A', 'A']`

- agent 0: answer `A`, correct `True`, family `theorem_property_application` / `theorem_property_application`
  - trace excerpt: 
- agent 1: answer `A`, correct `True`, family `theorem_property_application` / `theorem_property_application`
  - trace excerpt: 
- agent 2: answer `A`, correct `True`, family `symbolic_formulation` / `algebraic_derivation`
  - trace excerpt: 
- agent 3: answer `A`, correct `True`, family `decomposition` / `decomposition`
  - trace excerpt: 
- agent 4: answer `A`, correct `True`, family `symbolic_formulation` / `algebraic_derivation`
  - trace excerpt: 

## bank_div

### sample 1: `78a253d607eb`

- gold: `A`; vote: `D`; vote_correct: `0`
- answers: `['D', 'D', 'D', 'D', 'D']`

- agent 0: answer `D`, correct `False`, family `backward_reasoning` / `backward_reasoning`
  - trace excerpt: 
- agent 1: answer `D`, correct `False`, family `backward_reasoning` / `backward_reasoning`
  - trace excerpt: 
- agent 2: answer `D`, correct `False`, family `theorem_property_application` / `definition_application`
  - trace excerpt: 
- agent 3: answer `D`, correct `False`, family `backward_reasoning` / `consistency_verification`
  - trace excerpt: 
- agent 4: answer `D`, correct `False`, family `case_analysis` / `case_analysis`
  - trace excerpt: 

### sample 2: `6d9f2826542d`

- gold: `B`; vote: `B`; vote_correct: `1`
- answers: `['B', 'B', 'A', 'B', 'D']`

- agent 0: answer `B`, correct `True`, family `case_analysis` / `case_analysis`
  - trace excerpt: 
- agent 1: answer `B`, correct `True`, family `case_analysis` / `comparative_reasoning`
  - trace excerpt: 
- agent 2: answer `A`, correct `False`, family `case_analysis` / `option_elimination`
  - trace excerpt: 
- agent 3: answer `B`, correct `True`, family `case_analysis` / `option_elimination`
  - trace excerpt: 
- agent 4: answer `D`, correct `False`, family `option_elimination` / `option_elimination`
  - trace excerpt: 

## shared_baseline

### sample 1: `d31d60ce0f92`

- gold: `D`; vote: `D`; vote_correct: `1`
- answers: `['D', 'D', 'D', 'D', 'D']`

- agent 0: answer `D`, correct `True`, family `case_analysis` / `option_elimination`
  - trace excerpt: 
- agent 1: answer `D`, correct `True`, family `case_analysis` / `case_analysis`
  - trace excerpt: 
- agent 2: answer `D`, correct `True`, family `comparative_reasoning` / `comparative_reasoning`
  - trace excerpt: 
- agent 3: answer `D`, correct `True`, family `case_analysis` / `comparative_reasoning`
  - trace excerpt: 
- agent 4: answer `D`, correct `True`, family `comparative_reasoning` / `comparative_reasoning`
  - trace excerpt: 

### sample 2: `b0910f2a6434`

- gold: `A`; vote: `A`; vote_correct: `1`
- answers: `['A', 'A', 'A', 'A', 'A']`

- agent 0: answer `A`, correct `True`, family `theorem_property_application` / `theorem_property_application`
  - trace excerpt: 
- agent 1: answer `A`, correct `True`, family `proof_by_contradiction` / `consistency_verification`
  - trace excerpt: 
- agent 2: answer `A`, correct `True`, family `proof_by_contradiction` / `consistency_verification`
  - trace excerpt: 
- agent 3: answer `A`, correct `True`, family `theorem_property_application` / `consistency_verification`
  - trace excerpt: 
- agent 4: answer `A`, correct `True`, family `theorem_property_application` / `theorem_property_application`
  - trace excerpt: 

## bank_baseline

### sample 1: `7b933e5ce692`

- gold: `A`; vote: `A`; vote_correct: `1`
- answers: `['A', 'A', 'D', 'A', 'A']`

- agent 0: answer `A`, correct `True`, family `option_elimination` / `option_elimination`
  - trace excerpt: 
- agent 1: answer `A`, correct `True`, family `option_elimination` / `consistency_verification`
  - trace excerpt: 
- agent 2: answer `D`, correct `False`, family `option_elimination` / `option_elimination`
  - trace excerpt: 
- agent 3: answer `A`, correct `True`, family `option_elimination` / `comparative_reasoning`
  - trace excerpt: 
- agent 4: answer `A`, correct `True`, family `option_elimination` / `comparative_reasoning`
  - trace excerpt: 

### sample 2: `3676de3e7760`

- gold: `B`; vote: `B`; vote_correct: `1`
- answers: `['B', 'B', 'B', 'B', 'B']`

- agent 0: answer `B`, correct `True`, family `backward_reasoning` / `backward_reasoning`
  - trace excerpt: 
- agent 1: answer `B`, correct `True`, family `backward_reasoning` / `option_elimination`
  - trace excerpt: 
- agent 2: answer `B`, correct `True`, family `theorem_property_application` / `definition_application`
  - trace excerpt: 
- agent 3: answer `B`, correct `True`, family `backward_reasoning` / `backward_reasoning`
  - trace excerpt: 
- agent 4: answer `B`, correct `True`, family `backward_reasoning` / `option_elimination`
  - trace excerpt: 
