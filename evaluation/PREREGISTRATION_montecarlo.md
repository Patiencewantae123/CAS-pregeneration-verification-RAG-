# Pre-registration: Study 4, Monte Carlo sensitivity analysis of the CAS selection layer

Written 2026-11-03, before any simulation was run. `evaluation/run_montecarlo.py` stores the SHA-256 of this file in its output.

## Why

Studies 1 and 2 tested CAS with two LLMs at one parameter setting: trust weights 0.35/0.25/0.20/0.20, τ = 0.45, and the DeBERTa-v3 detector at θ = 0.99. Reviewers can still ask three questions that more LLM runs on CPU cannot afford to answer:

- **Q1, parameter robustness.** Are the selection-level findings an artefact of the hand-set trust weights and τ?
- **Q2, provenance quality.** How reliable must source metadata be before provenance-based selection keeps poison out of the prompt? Does CAS use that metadata as well as a plain source filter does?
- **Q3, detector quality.** How good must the conflict detector be, at a given false-alarm rate, for CAS to stop poison reaching the prompt unannounced?

## What this study is and is not

- It simulates only the **selection layer**: which passages reach the prompt, and which are marked by a conflict note.
- It runs no LLM, so it says nothing directly about answers. Its endpoints are preconditions for poisoning, not poisoning itself.
- It runs no new model inference. BM25 scores and DeBERTa-v3 NLI predictions are the real ones from studies 1–2, read from `results/nli_cache.jsonl`. A missing prediction is an error; the model is never loaded.

## Data

- The 400 questions of studies 1 and 2 (`data/hotpotqa_poison.json`, `data/hotpotqa_poison2.json`), unchanged.
- θ = 0.99 throughout.
- A "true conflict pair" is a poison passage and the gold passage it was made from: the gold passage with the highest word overlap, the same rule as `run_detector_probe.py`.
- A "clean pair" is two original passages.

## Methods (on every simulated draw)

- Vanilla RAG
- BM25 top-3
- Provenance + BM25 top-3, the source filter: provenance ≥ 0.5, as in studies 1–2
- CAS w/o graph
- CAS-NLI penalty
- CAS-NLI preserve

The methods are a vectorised re-implementation of `run_poison.py` and `python/services/`. CAS-keyword is not simulated.

**Validation gate.** Before any simulation, the re-implementation must reproduce the stored selections of studies 1–2 exactly: every method, both conditions, all 400 questions, default parameters, and for CAS-NLI preserve the stored conflict pairs too. If any selection differs, the simulation does not run.

## Endpoints (per question and draw)

1. **Silent poison exposure (primary).** At least one poison passage is in the prompt and is not marked by a conflict note. A poison passage is marked when it is one end of a conflict pair the method reports. Only CAS-NLI preserve reports pairs.
2. **Any poison exposure.** At least one poison passage is in the prompt.
3. **Answer available.** At least one retained original passage contains the gold answer (same normalisation as `run_llm_generation.normalize`).
4. **Passages retained.**

## Experiments

Each experiment uses its own seeded generator (seed 42 + experiment number).

### E1: weights and threshold (Q1)

- 1,250 random settings: trust weights ~ Dirichlet(1, 1, 1, 1) and τ ~ U(0.30, 0.60), plus the default setting as a reference.
- Every setting runs on all 400 questions in condition A (provenance 0.5) and in condition B.
- In condition B, provenance is redrawn for every setting and question with the study's generator: trusted ~ U(0.55, 0.95), untrusted ~ U(0.05, 0.45), each passage flipped with probability 0.2.
- Runs: 1,251 × 400 × 2 = **1,000,800 question-level runs**.

### E2: provenance quality (Q2)

- Default weights and τ.
- Provenance is trusted ~ U(0.3 + s/2, 0.7 + s/2) and untrusted ~ U(0.3 − s/2, 0.7 − s/2), each passage flipped with probability f.
- Separation s ∈ {0, 0.1, …, 0.6}. Condition B is s = 0.5, f = 0.2; at s = 0 the metadata carries no signal.
- Flip rate f ∈ {0, 0.1, 0.2, 0.3, 0.4, 0.5}.
- 42 cells × 400 questions × 60 draws = **1,008,000 runs**.

### E3: detector quality (Q3)

- Default weights and τ; conditions A and B, with B provenance redrawn as in E1.
- A synthetic detector replaces the real contradiction labels:
  - Each true conflict pair is flagged with probability r ∈ {0, 0.1, …, 1.0}.
  - Each clean pair is flagged with a per-pair probability chosen so that the question-level false-alarm rate is F ∈ {0, 0.10, 0.25, 0.50}.
  - A flagged pair has confidence 0.99.
  - Every other pair keeps its real entailment or neutral label. A real contradiction label that the synthetic detector does not draw becomes neutral.
- The realised question-level recall and false-alarm rate are reported with each cell, so the cells can be placed against Fig. 14.
- 44 cells × 2 conditions × 400 questions × 30 draws = **1,056,000 runs**.

**Total: 3,064,800 question-level runs.** Each run evaluates all six methods.

### E4: does a conflict note matter? (exploratory, real LLM data)

- Data: the CAS-NLI preserve rows of studies 1–2 where poison reached the prompt.
- Compare poison adoption when the poison was marked by a conflict note against when it was not, per model, with counts and bootstrap CIs.
- This is observational: marked and unmarked cases are not randomised. It shows only whether "silent exposure" is a sensible proxy.

## Inference

- **The unit of inference is the question (n = 200 per study, 400 pooled), not the draw.** The number of draws is a simulation choice that removes Monte Carlo noise. It is not sample size, so no p-values are computed over draws.
- Each endpoint is first averaged over a question's draws. 95% CIs then come from 2,000 bootstrap resamples of questions.
- Results are reported per study and pooled.

## Pre-specified summaries

These summaries are fixed now; all are descriptive.

- **E1.** For CAS-NLI preserve against each comparator, per condition and endpoint, report:
  - the difference at the default setting;
  - the median and 5th–95th percentile of the difference across the 1,250 random settings;
  - the share of settings where CAS-NLI preserve is better;
  - the across-setting mean difference with its question-bootstrap CI.

  A finding is called **robust to the parameters** if the default-setting difference and at least 90% of random settings have the same sign.
- **E2.** For each flip rate f, report the smallest separation s (if any) at which each provenance-using method reaches silent exposure ≤ 0.05, and the smallest s (if any) at which CAS-NLI preserve is no worse than the source filter.
- **E3.** For each F and condition, report:
  - the smallest r (if any) at which CAS-NLI preserve's silent exposure is no higher than the best non-CAS comparator in that condition (BM25 top-3 in A; the source filter in B);
  - CAS-NLI penalty's answer availability at r = 0 and r = 1.
- **Figure.** Fig. 15 shows the primary endpoint for E1, E2 and E3 in full, whatever the results.
