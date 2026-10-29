# Pre-registration: CAS under poisoned retrieval (HotpotQA-Poison)

Written 2026-10-29, **before** the benchmark was built or any method was run on it. Its SHA-256 is recorded in
`results/poison_summary.json` by `run_poison.py`; any later edit to this file will be visible there and in git.
All results are reported, whether or not they support the hypotheses.

## Motivation

On clean benchmarks (HotpotQA, MuSiQue), CAS did not improve answer accuracy over Vanilla RAG, because the
context held nothing harmful to remove. CAS is designed for retrieval that contains conflicting or false
passages. This study tests it in that setting.

## Data

- Source: the official HotpotQA dev set (distractor setting, 7,405 questions; HF mirror `hotpotqa/hotpot_qa`).
- Sample: **200 questions**, drawn with seed 42 after these filters:
  - The answer is not yes/no and has at most 5 words.
  - The answer occurs, case-insensitively, in at least one supporting paragraph.
  - The question is not one of the 50 used in the earlier experiments, which were used to choose θ.
- Poison (entity substitution; Longpre et al. 2021, as in PoisonedRAG):
  - Each question gets a substitute answer: the answer of another sampled question with the same shape (contains a digit or not), drawn with seed 42. The substitute must not occur anywhere in the original pool.
  - For every supporting paragraph containing the answer, a copy is made with every occurrence of the answer replaced by the substitute.
  - The copy is inserted at a seeded random position in the pool of 10 original paragraphs.

## Conditions

- **A, no source metadata.** Every passage gets provenance 0.5, as in the earlier experiments.
- **B, noisy source metadata.**
  - Original paragraphs get provenance ~ U(0.55, 0.95); poison passages get ~ U(0.05, 0.45).
  - The metadata is noisy on purpose: each poison passage, independently with probability 0.2, instead gets a trusted-range value (a compromised source), and each original paragraph, with probability 0.2, gets an untrusted-range value.
  - Seed 42. These values are fixed here and will not be changed after results are seen.

## Methods (no parameter is tuned on this benchmark)

| Method | Settings |
|---|---|
| Vanilla RAG | All passages |
| BM25 top-3 | 3 highest BM25 |
| Provenance + BM25 top-3 (B only) | Drop passages with provenance < 0.5, then BM25 top-3 |
| CAS w/o graph | τ = 0.45 |
| CAS-keyword | τ = 0.45, original keyword verifier |
| CAS-NLI penalty | τ = 0.45, DeBERTa-v3 NLI, θ = 0.99 (chosen earlier by CV on other data) |
| **CAS-NLI preserve (primary)** | τ = 0.45, θ = 0.99, conflict notes passed to the LLM |

LLMs: Llama 3.2 3B and Gemma 3 4B via Ollama (temperature 0, seed 42). The QA prompt is the same as in the earlier experiments.

## Endpoints (computed on the first line of the answer, SQuAD normalisation)

- **Primary, poison adoption:** the answer contains the substitute. Lower is better.
- **Co-primary, accuracy:** the answer contains the original gold answer.
- Secondary (descriptive only): abstention; whether a poison passage or a gold passage reaches the context.

## Confirmatory hypotheses and test

- **H1 (A):** CAS-NLI preserve has lower poison adoption than Vanilla RAG and than BM25 top-3.
- **H2 (A):** its accuracy differs from Vanilla RAG and from BM25 top-3 (direction reported).
- **H3 (B):** the same two endpoints for CAS-NLI preserve vs Vanilla RAG, BM25 top-3, and Provenance + BM25 top-3.

Test: a two-sided paired sign-flip permutation (10,000 permutations) with a bootstrap 95% CI, and Holm
correction over all 20 confirmatory tests (2 models × 10).

## Decision rule (fixed in advance)

- **"CAS reduces poisoning"** is supported only if adoption is significantly lower after Holm correction, in the same direction for both models.
- **"CAS improves robustness"** additionally requires that accuracy is not significantly lower. A drop in adoption that comes with a significant accuracy loss (for example, from abstaining more) will be reported as a **trade-off, not an improvement**.
- **Under condition B, the claim must be stated relative to Provenance + BM25 top-3.** If CAS does not beat that baseline, the benefit is attributed to the metadata, not to CAS.
- All other comparisons are exploratory and labelled as such.
