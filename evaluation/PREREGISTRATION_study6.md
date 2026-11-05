# Pre-registration: study 6, a powered two-dataset test of CAS v2 on clean retrieval

Written 2026-11-05, before the study 6 samples were drawn and before any method was run on them. `run_study6.py` does not start until this file is committed to git with the content on disk; it records that commit and the SHA-256 of this file in `results/study6_summary.json`. Every result is reported.

## Why

Study 5 tested CAS v2 on 200 clean HotpotQA questions. v2 was more accurate than Vanilla RAG for both models (+0.065 and +0.055), but neither difference survived Holm correction (H9 not supported). It gave fewer wrong answers, significantly only for Llama 3.2 (H3 not supported). Study 5 also used one dataset only.

Study 6 asks the same questions with enough questions to answer them, on two datasets:
- **HotpotQA:** fresh questions; the dataset v2 was developed on.
- **MuSiQue:** answerable 2–4-hop questions with 20 paragraphs each, built to resist shortcut reasoning. v2 has never been run on MuSiQue answers, and nothing was tuned on it.

Study 6 is a new confirmatory study. It is not pooled with study 5, whose results stand as reported.

## What is fixed

- **CAS v2 is frozen** exactly as in study 5 (`evaluation/cas_v2.py`, unchanged since 2026-11-04: κ = 0.8, edits of 1–8 content words, provenance gate unused because there is no metadata, τ = 0.45, title links).
- **The datasets and methods were fixed before the pilot below was run.** The pilot only informed the sample sizes and the predictions.
- **Generation:** the study 1–5 QA prompt; Llama 3.2 3B and Gemma 3 4B; Ollama 0.40.13, temperature 0, seed 42, context 8,192 tokens.

## Data (`build_study6.py`)

- **HotpotQA:** N_HOTPOT questions of every type, from the dev set (distractor setting, 10 passages). Excluded: every question used, attacked or attempted in studies 1–5 and the earlier experiments. Drawn with seed 42.
- **MuSiQue:** N_MUSIQUE questions from the answerable dev set (`musique_ans_v1.0_dev.jsonl`, 2,417 questions, 20 paragraphs each). Excluded: the 200 development-check questions (`data/study6_dev_ids.json`) and the 20 questions of the earlier exploratory sample, by id and by question text. Five question texts occur twice in the dev set under different ids; only the first occurrence is eligible. Drawn with seed 42. Gold answers are the answer and its aliases.

## Methods

| Method | Selection | Answers generated |
|---|---|---|
| Vanilla RAG | every passage | yes |
| BM25 top-3 | 3 passages with the highest BM25 score | no (selection only) |
| Reranker top-3 | `cross-encoder/ms-marco-MiniLM-L-6-v2`, top 3 | yes |
| CAS v2 | as in study 5 | yes |

BM25 top-3 is left out of generation to bound CPU time; in study 5 the reranker was the stronger of the two top-3 filters in evidence F1.

## Endpoints (the definitions and scoring code of studies 3 and 5)

- **Accuracy:** a gold answer (or alias) appears in the first line of the response, as whole words; an abstention is never correct.
- **Wrong answer:** neither correct nor an abstention.
- **Evidence F1** per question, against the supporting paragraphs.
- **Descriptive only:** precision, recall, full-evidence rate, abstention, answer in context, prompt tokens, latency.

## Confirmatory hypotheses (CAS v2 minus comparator; per dataset)

| | Hypothesis | Tests per dataset | Family |
|---|---|---|---|
| H1 | v2 is more accurate than Vanilla RAG | 2 (per model) | primary |
| H2 | v2 gives fewer wrong answers than Vanilla RAG | 2 (per model) | primary |
| H3 | v2 is non-inferior to Vanilla RAG in accuracy: the 95% CI lower bound of the difference is above −0.05 | 2 CIs | — |
| H4 | v2 is more accurate than Reranker top-3 | 2 (per model) | secondary |
| H5 | v2 has higher evidence F1 than Reranker top-3 and than BM25 top-3 | 2 | secondary |

- **Test:** two-sided paired sign-flip permutation (10,000 permutations) on per-question differences; 95% percentile bootstrap CIs (2,000 resamples).
- **Multiplicity:** Holm correction within each family. The primary family has 8 tests (H1, H2 × 2 datasets × 2 models); the secondary family has 8 (H4, H5). H3 is judged by its CIs.
- **Decision rule (fixed in advance), per dataset:** a hypothesis is supported when every one of its tests is significant (Holm p < 0.05 within its family) in the predicted direction, for both models where tested per model. H3 is supported when it holds for both models.
- **Exploratory:** abstention vs Vanilla RAG and wrong answers vs the reranker (one Holm family); accuracy differences by question type (HotpotQA bridge/comparison) and hop count (MuSiQue), descriptive only.

PILOT_AND_PREDICTIONS
