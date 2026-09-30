# Provenance audit

No HotpotQA, HoVer, IFBench, PUPA, or Hendrycks MATH source data, project version/config, original-split manifest, or reference setup appears in the tracked repository at the base commit. Available-example counts are therefore **unknown**, not zero. No external dataset was downloaded. The proposed upstream identities below are references only and do not freeze a project choice.

| ID | Candidate upstream and task facts | Version/config and project source | Fields needed if selected | License/reference status |
| --- | --- | --- | --- | --- |
| hotpotqa | [Original HotpotQA source](https://hotpotqa.github.io/) offers question, answer, context and supporting facts; answer EM/F1 and support/joint metrics differ. | `PROVENANCE_REQUIRES_SCIENTIFIC_DECISION`; distractor/fullwiki and answer-only/joint unselected | ID, question, answer; context/support if selected | Check selected upstream release |
| hover | [Original HoVer source](https://hover-nlp.github.io/) defines claim verification plus ranked supporting facts and a joint HoVer score. | `PROVENANCE_REQUIRES_SCIENTIFIC_DECISION`; task/corpus/retrieval scope unselected | ID, claim, label, evidence/document ranking | Check selected upstream release |
| ifbench | [Official IFBench](https://github.com/allenai/IFBench) reports prompt-level loose accuracy and retains per-instruction checks. | `PROVENANCE_REQUIRES_SCIENTIFIC_DECISION`; dataset version and project constraint set unselected | key, prompt, instruction IDs, kwargs | Official code Apache 2.0; data ODC-BY-1.0 |
| pupa | [PAPILLON authors' repository](https://github.com/Columbia-NLP-Lab/PAPILLON) describes PUPA with user-LLM interactions and privacy-sensitive content. | `PROVENANCE_REQUIRES_SCIENTIFIC_DECISION`; release and task unselected | Cannot freeze without task/privacy protocol | Check selected release and handling policy |
| math | [Original MATH source](https://github.com/hendrycks/math) includes an answer-equivalence helper. | `PROVENANCE_REQUIRES_SCIENTIFIC_DECISION`; original/derivative version and scorer unselected | ID, problem, solution/boxed answer, subject/level if applicable | Check selected upstream release |

No content hash or manifest hash is claimed for real benchmark data. Synthetic fixtures used in tests carry a local SHA-256 only. This is intentionally not a formal dataset freeze.
