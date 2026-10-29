# Journal figures: CAS pre-generation verification

Every number in this folder is measured. The figures are drawn from `evaluation/results/results.json` and `llm_summary.json`, which the evaluation scripts produce by running the CAS code in `python/` on the datasets in `data/`. Nothing is typed in by hand. `make_all.py` rebuilds all ten figures and `tables.md`, then checks the plotted values against the results files (last run: **PASS**).

## Reproduce

```bash
py evaluation/run_evaluation.py         # ~45 min on CPU (DeBERTa NLI); predictions cached in nli_cache.jsonl
py evaluation/run_llm_generation.py     # needs Ollama with llama3.2 and gemma3; responses cached
py journal_figures/make_all.py          # figNN_*.png (600 dpi) + .pdf + data/figNN_*.csv + tables.md
```

Environment: Python 3.14, numpy 2.5, pandas 3.0, matplotlib 3.11, torch 2.15 (CPU), sentence-transformers, Ollama 0.40.12, Intel CPU, no GPU. Seed 42; 2,000 bootstrap resamples; 10,000 sign-flip permutations; Holm correction within each family of tests.

## Methods compared

All settings were fixed before any results were seen. τ = 0.45 is the `CASEngine` default; the NLI label is the argmax; the partner-relevance rule (0.5) was set once. None were tuned on these data.

| Label | What it does |
|---|---|
| Vanilla RAG | Keeps every candidate passage |
| BM25 top-3 | Keeps the 3 passages with the highest BM25 score |
| CAS w/o graph | CAS trust score (relevance, provenance prior 0.5, rank), no NLI |
| CAS-keyword | The original CAS: keyword "NLI" plus contradiction penalty |
| CAS-NLI | `cross-encoder/nli-deberta-v3-base` (symmetric, two directions) plus contradiction penalty |
| CAS-NLI preserve | DeBERTa NLI with no penalty. When a kept passage contradicts another passage of relevance ≥ 0.5, both are kept and the conflict is labelled for the LLM |

## What changed since the first version of these figures

1. **The NLI model replaced the keyword verifier** (`python/services/nli_model_verifier.py`).
2. **A conflict-preserving synthesis mode was added** (`TrustAwareSynthesizer.filter_preserving_conflicts`).
3. **Evaluation adapter fix:** relevance is now BM25 divided by the pool maximum. It was previously min-max normalised, which forced the weakest passage to 0 and so, for two-passage pools, always dropped one of them. All methods were re-run under the corrected adapter, which changed the earlier CAS-keyword numbers (for example HotpotQA F1 went from 0.580 to 0.552). The earlier numbers remain in git history.

## Figure → data mapping

| Figure | Script | Plotted data | Source |
|---|---|---|---|
| 1 | `fig01_precision_recall_tradeoff.py` | `data/fig01_*.csv` | `results.json` → `evidence_selection` |
| 2 | `fig02_paired_differences.py` | `data/fig02_*.csv` | `paired_tests` |
| 3 | `fig03_conflict_graph_ablation.py` | `data/fig03_*.csv` | `graph_diagnostics` |
| 4 | `fig04_conflict_hallucination_detection.py` | `data/fig04_*.csv` | `contradiction_by_group`, `hallucination` |
| 5 | `fig05_performance_cost.py` | `data/fig05_*.csv` | `timing`, `evidence_selection` |
| 6 | `fig06_threshold_sensitivity.py` | `data/fig06_*.csv` | `threshold_sweep` |
| 7 | `fig07_error_analysis.py` | `data/fig07_*.csv` | `error_analysis` |
| 8 | `fig08_llm_answer_quality.py` | `data/fig08_*.csv` | `llm_summary.json` → `qa` |
| 9 | `fig09_llm_grounding_conflicts.py` | `data/fig09_*.csv` | `llm_scores.csv`, `llm_conflict_scores.csv` |
| 10 | `fig10_llm_cost_tradeoff.py` | `data/fig10_*.csv` | `llm_scores.csv` |

Every table cell is in `tables.md` (Tables 1–10).

## Captions

**Fig. 1. Evidence-selection precision and recall.** Means with 95% bootstrap CIs on both axes; dotted curves are iso-F1 contours.
- (a) HotpotQA, n = 50 questions, 10 candidates each. CAS-NLI preserve: P 0.650, R 0.700, F1 0.630. CAS w/o graph: F1 0.649. BM25 top-3: 0.592. CAS-keyword: 0.552. CAS-NLI: 0.349 (P 0.421, R 0.330).
- (b) MuSiQue, n = 15 questions, 20 candidates, at least one gold paragraph each. CAS-NLI preserve F1 0.401; CAS-NLI 0.239.

**Fig. 2. Paired per-question differences for the two CAS-NLI variants.** 95% bootstrap CIs. Filled markers show Holm-adjusted p < 0.05 over all 72 evidence-selection tests.
- **CAS-NLI preserve** improves F1 over Vanilla RAG on HotpotQA (+0.297 [0.231, 0.359], Holm p = 0.007) and over CAS-keyword (+0.079 [0.039, 0.123], Holm p = 0.010).
- Against BM25 top-3, its precision is higher (+0.156, Holm p = 0.007) but the F1 difference is not significant (+0.038 [−0.011, 0.086]).
- **CAS-NLI with the penalty** is significantly worse than the same pipeline without the graph (HotpotQA F1 −0.301 [−0.396, −0.205], Holm p = 0.007).

**Fig. 3. What each conflict graph does.**
- (a) Share of passage pairs labelled Entailment (E), Contradiction (C) or Neutral. The keyword verifier never fires a contradiction. DeBERTa labels 340 of 2,250 HotpotQA pairs (15.1%) and 273 of 2,850 MuSiQue pairs (9.6%) as contradictions, almost all between unrelated or mutually consistent passages.
- (b) Mean agreement credit received by gold vs distractor passages.
- (c) Mean contradiction penalty: 1.07 (gold) vs 1.17 (distractor) on HotpotQA, 1.51 vs 1.42 on MuSiQue. The penalty does not separate evidence from noise.

**Fig. 4. Conflict and hallucination detection.**
- (a) Fraction of passage pairs flagged as contradictory, with Wilson 95% CIs. DeBERTa flags 60.7% of explicit and 31.8% of implicit WikiContradict contradictions, and 12.0% of consistent HotpotQA gold pairs (false positives). Pooled over 120 pairs: P 0.833, R 0.536, F1 0.652. The keyword verifier flags nothing (F1 0).
- (b) RAGTruth ROC, 12 of 50 summaries hallucinated. The unsupported-token rate gives AUROC 0.81 [0.69, 0.92]. Sentence-level DeBERTa entailment gives 0.55 [0.37, 0.74], which is not distinguishable from chance.
- Not shown: on MultiNLI, DeBERTa scores 20/20. This is **not** a valid result, because these samples are from the MultiNLI training set the model was trained on.

**Fig. 5. Evidence F1 vs selection latency per query** (log scale; mean ± SD over 20 repeats; NLI methods include CPU model time measured at first computation). DeBERTa costs about 23 s per HotpotQA query (90 directional pairs) and 84 s per MuSiQue query (380 pairs), against under 5 ms for every non-NLI method. The cost grows quadratically with the number of candidates.

**Fig. 6. Sensitivity to the trust threshold τ** (HotpotQA and MuSiQue pooled, n = 65). At τ = 0.45, F1 is 0.511 (CAS-keyword), 0.323 (CAS-NLI) and 0.578 (CAS-NLI preserve). CAS-NLI stays below 0.33 at every τ. Above 0.675, the most a passage can score without agreement credit, all variants collapse. τ was swept on the evaluation data, so this figure is descriptive, not a tuning result.

**Fig. 7. Where full-pipeline decisions at τ = 0.45 go wrong.**
- (a) Gold passages. The CAS-NLI penalty removes 37 of 100 HotpotQA gold passages and 10 of 30 MuSiQue gold passages. CAS-keyword loses gold passages mainly through low lexical relevance (22 of its 28 HotpotQA misses).
- (b) Distractor passages. 49 of the 104 HotpotQA distractors that CAS-keyword retains get through only via agreement credit. The NLI penalty removes 27 HotpotQA distractors, but at the cost of 37 gold passages.

**Fig. 8. End-to-end answer quality, two local LLMs × six evidence-selection methods.** 95% bootstrap CIs.
- On HotpotQA, Vanilla RAG remains the most accurate for both models: Llama 0.60, Gemma 0.56.
- CAS-NLI preserve reaches 0.52 (Llama) and 0.56 (Gemma); CAS-NLI with the penalty falls to 0.38 and 0.34.
- After Holm correction over 84 tests, no CAS variant is significantly better than any baseline. The only significant QA difference is a loss: CAS-NLI vs Vanilla, Gemma HotpotQA F1 −0.283 [−0.404, −0.165], Holm p = 0.008.
- MuSiQue (n = 15) intervals are too wide to separate the methods.

**Fig. 9. Grounding, abstention and conflict handling.**
- (a) Ungrounded answers. With the penalty, CAS-NLI leaves an **empty context** for 19 of 65 questions; the models then answer from memory, giving ungrounded rates of 0.35 (Llama) and 0.38 (Gemma). Every other method stays at or below 0.043, and CAS-NLI preserve never produces an empty context.
- (b) Abstentions. Every filter increases abstention (Llama: 0.12 for Vanilla, 0.20–0.40 for the filtered methods), because filtering sometimes removes the passage holding the answer.
- (c) WikiContradict conflict flagging (n = 50). Vanilla gives 0.48 (Llama) and 0.70 (Gemma); CAS-NLI preserve 0.44 and 0.44. The conflict notes do not raise flagging above simply showing both passages. Gemma's drop vs Vanilla, −0.26 [−0.44, −0.10], is not significant after Holm correction (p = 0.44).

**Fig. 10. Answer accuracy vs generation cost** (HotpotQA and MuSiQue pooled, n = 65 per point).
- Filtering cuts the mean prompt from about 1,750 tokens to 280–560 and CPU generation time from 13–16 s to 1.6–4.9 s.
- CAS-NLI preserve keeps Gemma's accuracy (0.477, the same as Vanilla) at 28% of the prompt length. For Llama it costs 0.12 accuracy (0.415 vs 0.538).

## Summary of evidence

| Claim | Supported? |
|---|---|
| A real NLI model detects contradictions that the keyword verifier misses | **Yes**: F1 0.652 vs 0 (Fig. 4a) |
| Penalising contradicting passages improves evidence selection | **No**: it removes 37% of gold evidence and sometimes all of it (Figs. 3, 7, 9a) |
| Conflict-preserving CAS-NLI selects better evidence than the original CAS and Vanilla RAG | **Yes** (Fig. 2) |
| …than plain BM25 top-3 | Higher precision only; F1 not significant (Fig. 2) |
| CAS improves LLM answer accuracy | **No** significant improvement with 3–4B models (Fig. 8) |
| CAS reduces prompt size and generation latency | **Yes**, about 3.5× shorter prompts (Fig. 10), but the NLI step itself costs 23–84 s per query on CPU (Fig. 5) |
| CAS improves conflict surfacing | **No**: showing both passages (Vanilla) does at least as well (Fig. 9c) |

## Limitations

- **Small samples.** n = 50 (HotpotQA), 15 (MuSiQue), 50 (WikiContradict), 50 (RAGTruth), so CIs are wide.
- **Small models.** Only 3–4B local LLMs were tested; larger models may use conflict notes differently.
- **Automatic answer metrics.** String match and keyword cues; a human-annotated sample of `llm_generations.jsonl` is needed to validate them.
- **No source metadata.** Provenance is a constant prior (0.5), because the benchmarks carry no source reliability. CAS's trust component is therefore only partly exercised.
- **Self-RAG and CRAG are not implemented.** The comparison numbers previously in `results_cas.py` and the dashboards were not measured, and must not be cited.

## Audit of figures previously in the repository

`cas_graphs_without_legend/`, `cas_graphs_without_legend.zip` and `extracted_graphs/*.png` were generated by `results_cas.py` from hardcoded values. They include two datasets not in `data/` (RGB, MultiDoc2Dial) and a baseline CPI that contradicts `compute_cpi` (77.5 vs 75.53). **Do not submit them.** `extracted_graphs/mergered.py` cannot run (missing imports and definitions).
