# Pre-registration: study 6, a powered two-dataset test of CAS v2 on clean retrieval

Written 2026-11-05, before the study 6 samples were drawn and before any method was run on them. `run_study6.py` does not start until this file is committed to git with the content on disk. It records that commit and the SHA-256 of this file in `results/study6_summary.json`. Every result is reported.

## Why

Study 5 tested CAS v2 on 200 clean HotpotQA questions, on one dataset only.
- v2 was more accurate than Vanilla RAG for both models (+0.065 and +0.055), but neither difference survived Holm correction (H9 not supported).
- It gave fewer wrong answers, significantly only for Llama 3.2 (H3 not supported).

At n = 200, the power to detect a gain of that size at the strictest Holm threshold was about 0.2–0.35 (see "Sample sizes" below).

Study 6 asks the same questions with enough questions to answer them, on two datasets:
- **HotpotQA:** fresh questions; the dataset v2 was developed on.
- **MuSiQue:** answerable 2–4-hop questions with 20 paragraphs each, built to resist shortcut reasoning. Nothing in v2 was tuned on MuSiQue.

Study 6 is a new confirmatory study. It is not pooled with study 5, whose results stand as reported.

## What was fixed, and when

1. **Before any method was run on MuSiQue** (`dev_study6.py` was written with these choices; only the dataset's format and hop counts had been inspected), the following were fixed:
   - the two datasets;
   - the comparison of CAS v2 with Vanilla RAG, BM25 top-3 and Reranker top-3;
   - CAS v2 itself, frozen exactly as in study 5 (`evaluation/cas_v2.py`, unchanged since 2026-11-04: κ = 0.8, edits of 1–8 content words, τ = 0.45, title links; the provenance gate is unused because there is no metadata).
2. **Then the pilot was run** (below). It showed that v2 loses evidence and accuracy on MuSiQue.
3. **After the pilot, three ways to recover that evidence were tried on development data only** (`dev_study6_variants.py`, below).
   - Two were not adopted.
   - The third, an abstention fallback, is added here as a second arm, *CAS v2 + fallback*, with its own hypothesis (H6).
   - It was chosen after seeing development results, so study 6 is its first test.
4. **Unchanged after the pilot:** v2, the datasets and the hypotheses H1–H5.

**Generation:**
- the study 1–5 QA prompt;
- Llama 3.2 3B and Gemma 3 4B (model IDs `a80c4f17acd5`, `a2af6cc3eb7f`);
- temperature 0, seed 42, context 8,192 tokens.

**Ollama runtime: 0.40.13**, the version of studies 3 and 5.
- Version 0.40.14 has been downloaded by Ollama's updater since 2026-11-03. It installs itself when the Ollama app next starts.
- `run_study6.py` checks the version before every question and stops if it is not 0.40.13. Cached answers are kept, and a stopped run is resumed only on 0.40.13. No study 6 answer can therefore come from another version.

## Data (`build_study6.py`)

- **HotpotQA:** 700 questions of every type, from the dev set (distractor setting, 10 passages), drawn with seed 42.
  - Excluded: every question used, attacked or attempted in studies 1–5 and the earlier experiments.
- **MuSiQue:** 200 questions from the answerable dev set (`musique_ans_v1.0_dev.jsonl`, 2,417 questions, 20 paragraphs each), drawn with seed 42.
  - Excluded, by id and by question text: the 200 development-check questions (`data/study6_dev_ids.json`) and the 20 questions of the earlier exploratory sample.
  - Five question texts occur twice in the dev set under different ids; only the first occurrence is eligible.
  - Gold answers are the answer and its aliases.

## Methods

| Method | Passages given to the model | Answers generated |
|---|---|---|
| Vanilla RAG | every passage | yes |
| BM25 top-3 | the 3 with the highest BM25 score | no (selection only) |
| Reranker top-3 | `cross-encoder/ms-marco-MiniLM-L-6-v2`, top 3 | yes |
| CAS v2 | as in study 5 | yes |
| CAS v2 + fallback | v2's answer. If the model abstains, it is asked again with every passage that passed v2's gate and quarantine, and that answer is scored. Its cost counts both calls. | yes (the second call only after an abstention) |

BM25 top-3 is left out of generation to bound CPU time. In study 5, the reranker was the stronger of the two top-3 filters in evidence F1.

## Endpoints (the definitions and scoring code of studies 3 and 5)

- **Accuracy:** a gold answer (or alias) appears in the first line of the response, as whole words; an abstention is never correct.
- **Wrong answer:** neither correct nor an abstention.
- **Evidence F1** per question, against the supporting paragraphs.
- **Descriptive only:** precision, recall, full-evidence rate, abstention, answer in context, fallback use, prompt tokens, latency.

## Confirmatory hypotheses (per dataset)

| | Hypothesis | Tests per dataset | Family |
|---|---|---|---|
| H1 | CAS v2 is more accurate than Vanilla RAG | 2 (per model) | primary |
| H2 | CAS v2 gives fewer wrong answers than Vanilla RAG | 2 (per model) | primary |
| H3 | CAS v2 is non-inferior to Vanilla RAG in accuracy: the 95% CI lower bound of the difference is above −0.05 | 2 CIs | — |
| H4 | CAS v2 is more accurate than Reranker top-3 | 2 (per model) | secondary |
| H5 | CAS v2 has higher evidence F1 than Reranker top-3 and than BM25 top-3 | 2 | secondary |
| H6 | CAS v2 + fallback is more accurate than Vanilla RAG | 2 (per model) | secondary |

- **Test:** two-sided paired sign-flip permutation (10,000 permutations) on per-question differences, with 95% percentile bootstrap CIs (2,000 resamples).
- **Multiplicity:** Holm correction within each family.
  - Primary family: 8 tests (H1 and H2 × 2 datasets × 2 models).
  - Secondary family: 12 tests (H4, H5 and H6).
  - H3 is judged by its CIs.
- **Decision rule (fixed in advance), per dataset:**
  - A hypothesis is supported when every one of its tests is significant (Holm p < 0.05 within its family) in the predicted direction, for both models where tested per model.
  - H3 is supported when it holds for both models.
- **Exploratory:**
  - One Holm family: abstention and wrong answers of v2 vs Vanilla RAG and the reranker; wrong answers of v2 + fallback vs Vanilla RAG; accuracy of v2 + fallback vs v2.
  - Descriptive only: accuracy differences by question type (HotpotQA bridge/comparison) and by hop count (MuSiQue).

## Development before this plan

### Pilot (`dev_study6.py`, output `results/study6_dev.json`)

**Selection**, on 200 MuSiQue dev questions excluded from study 6:

| Method | Precision | Recall | F1 | Every supporting paragraph kept | Paragraphs kept |
|---|---|---|---|---|---|
| Vanilla RAG | 0.132 | 1.000 | 0.231 | 100% | 20.0 |
| BM25 top-3 | 0.383 | 0.458 | 0.411 | 8.5% | 3.0 |
| Reranker top-3 | 0.458 | 0.548 | 0.491 | 18.0% | 3.0 |
| CAS v2 | 0.355 | 0.719 | 0.444 | 43.5% | 6.4 |

On HotpotQA, by comparison, v2 kept every supporting passage for 91% of study 5's questions. Its quarantine fired on 19% of these MuSiQue questions but never removed a supporting paragraph. The evidence is lost at the relevance threshold: later hops share few words with the question.

**Answers**, on the first 60 of these questions (unadjusted):

| Model | Vanilla RAG | CAS v2 | v2 − Vanilla RAG, accuracy | v2 − Vanilla RAG, wrong answers |
|---|---|---|---|---|
| Llama 3.2 | 0.267 | 0.133 | −0.133 [−0.233, −0.033] | 0.000 [−0.117, +0.117] |
| Gemma 3 | 0.167 | 0.150 | −0.017 [−0.117, +0.083] | −0.117 [−0.234, 0.000] |

### Variants tried after the pilot (`dev_study6_variants.py`, output `results/study6_variants_dev.json`)

- **Hop expansion: not adopted.** Rounds of BM25 search from the passages just added raised MuSiQue full evidence from 43.5% to at most 82%. But it got there by keeping 15.6 of 20 paragraphs. It also lowered HotpotQA evidence F1, from 0.754 to 0.645 even with one round (first 200 study 3 questions).
- **Cross-encoder chain: not adopted.** Iterative cross-encoder selection, with title links, reached at most 51% MuSiQue full evidence, against 43.5% for v2.
  - On HotpotQA it raised evidence F1 (best 0.809 vs 0.754), but not full evidence (0.88–0.89 vs 0.875).
  - It cost 0.3–0.6 s of selection per question, against v2's 3 ms.
  - It did not fix MuSiQue, which was its purpose.
- **Abstention fallback: adopted as the second arm.** All rows are unadjusted development results.

  | Data | Model | Vanilla RAG | v2 | v2 + fallback | Fallback − Vanilla RAG, accuracy |
  |---|---|---|---|---|---|
  | HotpotQA, study 5 clean (200, analysed) | Llama 3.2 | 0.485 | 0.550 | 0.560 | +0.075 [+0.025, +0.130] |
  | | Gemma 3 | 0.395 | 0.450 | 0.465 | +0.070 [+0.015, +0.120] |
  | MuSiQue pilot (60) | Llama 3.2 | 0.267 | 0.133 | 0.200 | −0.067 [−0.150, +0.017] |
  | | Gemma 3 | 0.167 | 0.150 | 0.217 | +0.050 [−0.033, +0.133] |

  The fallback also raised wrong answers relative to v2 (for example, HotpotQA with Gemma 3: 0.440 → 0.495).

## Sample sizes

- **HotpotQA, n = 700.** The SD of the per-question accuracy difference (v2 − Vanilla RAG) in study 5 was 0.389 for Llama 3.2 and 0.403 for Gemma 3. At n = 700, the normal-approximation power at the strictest Holm threshold (α = 0.05/8, two-sided) is:
  - 0.84 (Llama 3.2) and 0.81 (Gemma 3) for a true difference of +0.055;
  - 0.75 and 0.71 for +0.05.
- **MuSiQue, n = 200.** This is a generalisation test, not a test powered for a small gain. A difference of the pilot's size for Llama 3.2 (−0.133) would be detected with near certainty.
- **CPU budget:** about 12 hours, from the pilot's timings.

## Predictions, stated in advance

- **HotpotQA:**
  - H1, H3, H5 and H6 expected to be supported;
  - H4 expected (study 5, exploratory: +0.105 and +0.095 against the reranker);
  - H2 uncertain for Gemma 3.
- **MuSiQue:**
  - H1, H2 and H3 expected *not* to be supported. The pilot found v2 less accurate than Vanilla RAG for Llama 3.2.
  - H5 expected *not* to be supported. In the pilot, v2's F1 was below the reranker's.
  - H4 uncertain: v2 kept the full evidence far more often than the reranker, at lower precision.
  - H6 uncertain.
