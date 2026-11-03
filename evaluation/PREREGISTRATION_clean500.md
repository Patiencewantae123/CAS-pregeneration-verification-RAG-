# Pre-registration: study 3, clean-retrieval replication at scale (HotpotQA-500)

Written 2026-11-03, before the sample was drawn or any method was run on it. The SHA-256 of this file is
recorded in `results/clean500_summary.json` by `run_clean500.py`. Every result is reported.

## Why

The clean-retrieval results in Figs. 1–10 rest on 50 HotpotQA and 15 MuSiQue questions, so their CIs are wide. This
study repeats the comparison at n = 500 and adds a stronger relevance baseline, a cross-encoder reranker.

## Data

- Source: the HotpotQA dev set (distractor setting; `data/external/hotpot_dev_distractor.parquet`).
- Sample: **500 questions**, drawn with seed 42 from all question types.
- Excluded: the 50 questions of the earlier experiments and the 400 questions of the two poisoning studies.
- Each question keeps its original 10 paragraphs. Gold evidence is the set of supporting-fact paragraphs.

## Methods (nothing is tuned on this sample)

| Method | Settings |
|---|---|
| Vanilla RAG | All 10 passages |
| BM25 top-3 | 3 highest BM25 |
| Reranker top-3 | `cross-encoder/ms-marco-MiniLM-L-6-v2`, 3 highest scores |
| CAS w/o graph | τ = 0.45, provenance 0.5 |
| CAS-keyword | τ = 0.45, original keyword verifier |
| CAS-NLI penalty | τ = 0.45, DeBERTa-v3 NLI, θ = 0.99 (chosen earlier by CV on other questions) |
| **CAS-NLI preserve (primary)** | τ = 0.45, θ = 0.99, conflict notes passed to the LLM |

- Evidence metrics are computed for all seven methods.
- LLM answers are generated for six of them; CAS-keyword is evidence-only, to bound CPU time.
- LLMs: Llama 3.2 3B and Gemma 3 4B (Ollama, temperature 0, seed 42), with the same QA prompt as before.

## Endpoints

- Evidence F1 per question, against the supporting paragraphs.
- Answer accuracy: the gold answer appears in the first line of the answer, after SQuAD normalisation.
- Descriptive only: precision, recall, prompt tokens, generation latency, abstention, ungrounded rate.

## Confirmatory hypotheses (primary = CAS-NLI preserve)

- **H1:** evidence F1 differs from Vanilla RAG, BM25 top-3 and Reranker top-3 (3 tests).
- **H2:** answer accuracy differs from Vanilla RAG, BM25 top-3 and Reranker top-3, per model (6 tests).
- **H3, non-inferiority:** the accuracy difference vs Vanilla RAG has a 95% bootstrap CI lower bound above −0.05, per model. This is the claim that filtering costs no meaningful accuracy.

Test: a two-sided paired sign-flip permutation (10,000 permutations), with Holm correction over the 9 tests of H1 and H2. H3 is judged by its CI against the pre-set margin of 0.05.

## Decision rule (fixed in advance)

- "CAS selects better evidence" requires H1 significant and favouring CAS against all three comparators.
- "CAS improves answers" requires H2 significant and favouring CAS against a comparator for both models.
- "CAS filtering is accuracy-neutral" requires H3 to hold for both models.
- Everything else, including CAS-NLI preserve vs CAS-NLI penalty, is exploratory.
