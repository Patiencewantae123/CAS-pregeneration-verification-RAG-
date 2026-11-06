# Pre-registration: study 7, CAS v2 with a larger generator (Llama 3.1 8B)

Written 2026-11-06, before the study 7 sample was drawn and before any method was run on it. `run_study7.py` does not start until this file is committed to git with the content on disk. It records that commit and the SHA-256 of this file in `results/study7_summary.json`. Every result is reported.

## Why

Studies 5 and 6 tested CAS v2 with two small generators, Llama 3.2 3B and Gemma 3 4B.
- In study 6 (HotpotQA, n = 700), v2 was more accurate than Vanilla RAG (+0.047 and +0.059) and gave fewer wrong answers (−0.083 and −0.107). All four tests survived Holm correction.
- An exploratory analysis (manuscript Section 5.10) traced most of the accuracy gain to reading: on questions where the gold answer was in both prompts, v2 was more accurate by +0.086 and +0.112. v2 works, in part, by removing distracting passages.
- A larger model may be less distracted, so the gain may shrink. In the exploratory clean benchmark (manuscript Section 5.1; 50 HotpotQA questions), Vanilla RAG was the most accurate method for Llama 3.1 8B (0.68), as it was for the small models. CAS v2 has never been run with this model.

Study 7 asks whether v2's HotpotQA result holds for an 8B generator. It is a new confirmatory study on fresh questions. It is not pooled with studies 5 or 6, whose results stand as reported.

## What is fixed

- **CAS v2**, frozen exactly as in studies 5 and 6 (`evaluation/cas_v2.py`, unchanged since 2026-11-04: κ = 0.8, edits of 1–8 content words, τ = 0.45, title links; the provenance gate is unused because there is no metadata).
- **The abstention fallback**, exactly as in study 6.
- **The methods, prompt, endpoints, tests and scoring code** of study 6. `run_study7.py` imports study 6's selection and answering functions unchanged.
- **Nothing was tuned for this study.** No study 7 question has been seen. No method has been run with Llama 3.1 8B since the exploratory clean benchmark.

**Generation:**
- the study 1–6 QA prompt;
- Llama 3.1 8B (`llama3.1:8b`), the only generator. Its weights file is `sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29`, pulled on 2026-10-29. `run_study7.py` checks through Ollama's API that the model is backed by exactly this file, and stops otherwise. (Llama 3.1 8B also wrote the study 5 attack texts. Study 7 has no attack.)
- temperature 0, seed 42, context 8,192 tokens, at most 96 output tokens (the study 1–6 options).

**Ollama runtime.** Studies 3, 5 and 6 ran on Ollama 0.40.13. Version 0.40.14 has been downloaded by Ollama's updater since 2026-11-03 and installs itself when the Ollama app next starts. The runtime for study 7 is therefore fixed as follows:
- The Ollama version that answers the first study 7 question is recorded in `results/study7_runtime.json`, with the model's weights file and API digest.
- `run_study7.py` checks the version before every question and stops if it changes. Cached answers are kept, and a stopped run is resumed only on the recorded version. Every study 7 answer therefore comes from one version.
- Studies 6 and 7 may differ in Ollama version as well as model. The comparison between them is descriptive only (below).

## Data (`build_study7.py`)

- **HotpotQA:** 1,000 questions of every type from the dev set (distractor setting, 10 passages), drawn with seed 42.
- **Excluded:** every question used, attacked or attempted in studies 1–6 and the earlier experiments, by question text, and study 6's sample also by id. This leaves 5,425 of the 7,405 dev questions eligible.
- **MuSiQue is not included.** In study 6, v2 did not raise accuracy on MuSiQue, because it loses evidence at selection, before any model is involved. A larger generator cannot change that, and the CPU time is better spent on a larger HotpotQA sample.

## Methods (as in study 6)

| Method | Passages given to the model | Answers generated |
|---|---|---|
| Vanilla RAG | all 10 passages | yes |
| BM25 top-3 | the 3 with the highest BM25 score | no (selection only) |
| Reranker top-3 | `cross-encoder/ms-marco-MiniLM-L-6-v2`, top 3 | yes |
| CAS v2 | as in studies 5 and 6 | yes |
| CAS v2 + fallback | v2's answer. If the model abstains, it is asked again with every passage that passed v2's gate and quarantine, and that answer is scored. Its cost counts both calls. | yes (the second call only after an abstention) |

## Endpoints (the definitions and scoring code of studies 3, 5 and 6)

- **Accuracy:** a gold answer appears in the first line of the response, as whole words; an abstention is never correct.
- **Wrong answer:** neither correct nor an abstention.
- **Net score:** correct = +1, abstention = 0, wrong = −1, per question (accuracy minus wrong answers).
- **Evidence F1** per question, against the supporting paragraphs.
- **Descriptive only:** precision, recall, full-evidence rate, abstention, exact match, token F1, selective accuracy, answer in context, fallback use, prompt tokens, latency.

## Confirmatory hypotheses

| | Hypothesis | Tests | Family |
|---|---|---|---|
| H1 | CAS v2 is more accurate than Vanilla RAG | 1 | primary |
| H2 | CAS v2 gives fewer wrong answers than Vanilla RAG | 1 | primary |
| H3 | CAS v2 is non-inferior to Vanilla RAG in accuracy: the 95% CI lower bound of the difference is above −0.05 | 1 CI | — |
| H4 | CAS v2 is more accurate than Reranker top-3 | 1 | secondary |
| H5 | CAS v2 has higher evidence F1 than Reranker top-3 and than BM25 top-3 | 2 | secondary |
| H6 | CAS v2 + fallback is more accurate than Vanilla RAG | 1 | secondary |
| H7 | CAS v2 has a higher net score than Vanilla RAG | 1 | secondary |

H7 is new. In study 6 it was an exploratory, post hoc analysis (manuscript Section 5.10); here it is tested in advance.

- **Test:** two-sided paired sign-flip permutation (10,000 permutations) on per-question differences, with 95% percentile bootstrap CIs (2,000 resamples).
- **Multiplicity:** Holm correction within each family. Primary family: 2 tests. Secondary family: 5 tests. H3 is judged by its CI.
- **Decision rule (fixed in advance):** a hypothesis is supported when every one of its tests is significant (Holm p < 0.05 within its family) in the predicted direction. H3 is supported when its CI lower bound is above −0.05.
- **Exploratory (one Holm family of 7 tests):** abstention of v2 vs Vanilla RAG; wrong answers of v2 vs the reranker; wrong answers, accuracy (vs v2) and net score of v2 + fallback; exact match and token F1 of v2 vs Vanilla RAG.
- **Descriptive only:**
  - v2 minus Vanilla RAG by question type (bridge, comparison);
  - the size of each effect next to its study 6 estimate. Studies 6 and 7 differ in model, sample and possibly Ollama version, so this comparison is not a test of model size.

## Smoke test (before this plan is committed)

`run_study7.py --smoke` runs every arm on 3 study 5 questions, which were already used and are excluded from study 7. It checks that the 8B model fits in memory and measures the time per question. It writes only `results/study7_smoke.json` and never reads or writes the study 7 sample.

## Sample size

- **n = 1,000.** In study 6 (HotpotQA), the SD of the per-question difference (v2 − Vanilla RAG) was 0.45–0.46 for accuracy and 0.44–0.45 for wrong answers.
- With those SDs, the normal-approximation power at the strictest Holm threshold of the primary family (α = 0.05/2, two-sided) is:
  - 0.88–0.91 for a true difference of 0.05;
  - 0.69–0.74 for 0.04;
  - 0.43–0.47 for 0.03.
- At n = 700, the study 6 size, power for 0.05 would be 0.73–0.78. The larger sample is chosen because the effect may be smaller with an 8B model.
- **CPU budget: about 11 hours.** This is a rough extrapolation, not a measurement: Llama 3.2 3B needed 15.4 s per question for the three answer arms in study 6, and an 8B model is about 2.5 times larger. The smoke test measures it.

## Predictions, stated in advance

- **H5 expected to be supported.** Selection does not depend on the generator, and v2's evidence F1 beat both top-3 filters in studies 5 and 6.
- **H4 expected to be supported.** In study 6, the reranker dropped a supporting passage for 41% of HotpotQA questions, and a missing passage costs any generator.
- **H3 expected to be supported.**
- **H1, H2, H6 and H7 uncertain.** A larger model may be less distracted by the passages v2 removes, which would shrink the gain in H1. It may also abstain less when v2 has dropped a needed passage, which would weaken H2.
