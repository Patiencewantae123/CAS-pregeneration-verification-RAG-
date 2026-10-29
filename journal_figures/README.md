# Journal figures: CAS pre-generation verification

Every number in this folder is measured. The figures are drawn from `evaluation/results/results.json`, `cv_thresholds.json` and `llm_summary.json`, which the evaluation scripts produce by running the CAS code in `python/` on the datasets in `data/`. Nothing is typed in by hand. `make_all.py` rebuilds all eleven figures and `tables.md`, then checks the plotted values against the results files (last run: **PASS, 11 figures**).

## Reproduce

```bash
py evaluation/run_evaluation.py         # ~45 min on CPU (DeBERTa NLI); predictions cached in nli_cache.jsonl
py evaluation/run_llm_generation.py --models llama3.2:latest gemma3:latest llama3.1:8b
                                        # Ollama, CPU only; ~3.5 h for the 8B model; responses cached
py journal_figures/make_all.py          # figNN_*.png (600 dpi) + .pdf + data/figNN_*.csv + tables.md
```

Environment:
- Software: Python 3.14, numpy 2.5, pandas 3.0, matplotlib 3.11, torch 2.15 (CPU), sentence-transformers, Ollama 0.40.12.
- Hardware: Intel CPU, 15 GB RAM, **no GPU**. Llama 3.1 8B ran at about 48 prompt tokens/s and 9 generated tokens/s.
- Statistics: seed 42; 2,000 bootstrap resamples; 10,000 sign-flip permutations; Holm correction within each family of tests.

## Methods compared

The fixed settings were chosen before any results were seen and were not tuned on these data:
- τ = 0.45, the `CASEngine` default.
- Partner-relevance rule = 0.5.

The only tuned quantity is the NLI contradiction threshold θ, used by the two θ-CV methods. θ is chosen by **5-fold cross-validation**:
- Folds are stratified by dataset (seed 42).
- The grid is {0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99}, and ties go to the larger θ.
- Each question is scored with the θ chosen on the *other* four folds, so no question influences its own threshold.

| Label | What it does |
|---|---|
| Vanilla RAG | Keeps every candidate passage |
| BM25 top-3 | Keeps the 3 passages with the highest BM25 score |
| CAS w/o graph | CAS trust score (relevance, provenance prior 0.5, rank), no NLI |
| CAS-keyword | The original CAS: keyword "NLI" plus contradiction penalty |
| CAS-NLI | `cross-encoder/nli-deberta-v3-base` (symmetric, two directions), argmax label, plus contradiction penalty |
| CAS-NLI preserve | DeBERTa NLI with no penalty. When a kept passage contradicts another passage of relevance ≥ 0.5, both are kept and the conflict is labelled for the LLM |
| CAS-NLI θ-CV | CAS-NLI, but a pair counts as a contradiction only if p(contradiction) ≥ θ, with θ chosen by CV |
| CAS-NLI θ-CV preserve | CAS-NLI preserve with the CV-chosen θ |

## What changed since the first version of these figures

1. **The NLI model replaced the keyword verifier** (`python/services/nli_model_verifier.py`).
2. **A conflict-preserving synthesis mode was added** (`TrustAwareSynthesizer.filter_preserving_conflicts`).
3. **Evaluation adapter fix:** relevance is now BM25 divided by the pool maximum. It was previously min-max normalised, which forced the weakest passage to 0 and so, for two-passage pools, always dropped one of them. All methods were re-run under the corrected adapter, which changed the earlier CAS-keyword numbers (for example HotpotQA F1 went from 0.580 to 0.552). The earlier numbers remain in git history.
4. **θ is now chosen by held-out cross-validation** (`run_cv` in `evaluation/run_evaluation.py`; Fig. 11).
5. **A third, larger LLM was added:** Llama 3.1 8B, run on CPU.

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
| 11 | `fig11_cv_threshold.py` | `data/fig11_*.csv` | `results.json` → `cv_threshold` (also `cv_thresholds.json`) |

Every table cell is in `tables.md` (Tables 1–10).

## Captions

**Fig. 1. Evidence-selection precision and recall.** Means with 95% bootstrap CIs on both axes; dotted curves are iso-F1 contours.
- (a) HotpotQA, n = 50 questions, 10 candidates each.
  - CAS-NLI θ-CV preserve: P 0.651, R 0.700, F1 0.632.
  - CAS w/o graph: F1 0.649. BM25 top-3: 0.592. CAS-keyword: 0.552.
  - CAS-NLI θ-CV (penalty): 0.447. CAS-NLI with argmax labels: 0.349.
- (b) MuSiQue, n = 15 questions, 20 candidates, at least one gold paragraph each. F1: CAS-NLI θ-CV preserve 0.401; CAS-NLI θ-CV 0.316; CAS-NLI 0.239.

**Fig. 2. Paired per-question differences for the two θ-CV variants.** 95% bootstrap CIs. Filled markers show Holm-adjusted p < 0.05 over all 150 evidence-selection tests.
- **CAS-NLI θ-CV preserve** improves HotpotQA F1 over Vanilla RAG (+0.299 [0.233, 0.360], Holm p = 0.015) and over CAS-keyword (+0.080 [0.041, 0.124], Holm p = 0.024).
- Against BM25 top-3, its precision is higher (+0.158 [0.085, 0.229], Holm p = 0.015) but the F1 difference is not significant (+0.040 [−0.009, 0.087]).
- It is not distinguishable from CAS w/o graph (F1 −0.018, Holm p = 1.0). The conflict graph therefore adds no measurable selection benefit.
- **CAS-NLI θ-CV with the penalty** is still significantly worse than the same pipeline without the graph (HotpotQA F1 −0.203 [−0.295, −0.117], Holm p = 0.024).
- On MuSiQue (n = 15), the only significant difference is θ-CV preserve's precision gain over Vanilla RAG (+0.230, Holm p = 0.015).

**Fig. 3. What each conflict graph does.**
- (a) Share of passage pairs labelled Entailment (E), Contradiction (C) or Neutral.
  - The keyword verifier never fires a contradiction.
  - With argmax labels, DeBERTa labels 340 of 2,250 HotpotQA pairs (15.1%) and 273 of 2,850 MuSiQue pairs (9.6%) as contradictions. At the CV-chosen θ = 0.99, it labels 135 and 80.
  - These are almost all between unrelated or mutually consistent passages.
- (b) Mean agreement credit received by gold vs distractor passages.
- (c) Mean contradiction penalty: 1.07 (gold) vs 1.17 (distractor) on HotpotQA, 1.51 vs 1.42 on MuSiQue. The penalty does not separate evidence from noise.

**Fig. 4. Conflict and hallucination detection.**
- (a) Fraction of passage pairs flagged as contradictory, with Wilson 95% CIs.
  - DeBERTa flags 60.7% of explicit and 31.8% of implicit WikiContradict contradictions, and 12.0% of consistent HotpotQA gold pairs (false positives).
  - Pooled over 120 pairs: P 0.833, R 0.536, F1 0.652.
  - The keyword verifier flags nothing (F1 0).
  - For detection, CV chose θ = 0.5 in all five folds, the same as argmax.
- (b) RAGTruth ROC, 12 of 50 summaries hallucinated.
  - The unsupported-token rate gives AUROC 0.81 [0.69, 0.92].
  - Sentence-level DeBERTa entailment gives 0.55 [0.37, 0.74], which is not distinguishable from chance.
- Not shown: on MultiNLI, DeBERTa scores 20/20. This is **not** a valid result, because these samples are from the MultiNLI training set the model was trained on.

**Fig. 5. Evidence F1 vs selection latency per query.**
- Log scale; mean ± SD over 20 repeats. NLI methods include CPU model time, measured at first computation.
- DeBERTa costs about 23 s per HotpotQA query (90 directional pairs) and 84 s per MuSiQue query (380 pairs). Every non-NLI method takes under 5 ms.
- The cost grows quadratically with the number of candidates.

**Fig. 6. Sensitivity to the trust threshold τ** (HotpotQA and MuSiQue pooled, n = 65).
- At τ = 0.45, F1 is 0.511 (CAS-keyword), 0.323 (CAS-NLI) and 0.578 (CAS-NLI preserve).
- CAS-NLI stays below 0.33 at every τ.
- Above 0.675, the most a passage can score without agreement credit, all variants collapse.
- τ was swept on the evaluation data, so this figure is descriptive, not a tuning result.

**Fig. 7. Where full-pipeline decisions at τ = 0.45 go wrong.**
- (a) Gold passages. The CAS-NLI penalty removes 37 of 100 HotpotQA gold passages and 10 of 30 MuSiQue gold passages. CAS-keyword loses gold passages mainly through low lexical relevance (22 of its 28 HotpotQA misses).
- (b) Distractor passages. 49 of the 104 HotpotQA distractors that CAS-keyword retains get through only via agreement credit. The NLI penalty removes 27 HotpotQA distractors, but at the cost of 37 gold passages.

**Fig. 8. End-to-end answer quality, three local LLMs × six evidence-selection methods.**
- Models: Llama 3.2 3B, Gemma 3 4B and Llama 3.1 8B, all at temperature 0. Error bars are 95% bootstrap CIs.
- On HotpotQA, Vanilla RAG is the most accurate method for every model: 0.60, 0.56 and 0.68.
- The larger model does not change the ranking. With Llama 3.1 8B:
  - BM25 top-3 reaches 0.68.
  - CAS-keyword and CAS-NLI θ-CV preserve reach 0.60.
  - CAS-NLI θ-CV (penalty) reaches 0.50, and CAS-NLI reaches 0.36.
- After Holm correction over 210 tests, **no CAS variant is significantly better than any baseline for any model**. The significant QA differences are all losses for CAS-NLI with argmax labels:
  - Gemma, HotpotQA F1 vs Vanilla: −0.283.
  - Llama 3.1 8B, HotpotQA vs Vanilla: accuracy −0.32, F1 −0.366.
  - Llama 3.1 8B, HotpotQA vs BM25: accuracy −0.32, F1 −0.333.
  - All have Holm p = 0.021.
- MuSiQue (n = 15) intervals are too wide to separate the methods.

**Fig. 9. Grounding, abstention and conflict handling** (three models).
- (a) Ungrounded answers (HotpotQA + MuSiQue).
  - With the penalty, CAS-NLI leaves an **empty context** for 19 of 65 questions. The models then answer from memory, giving ungrounded rates of 0.35, 0.38 and 0.38 for the three models.
  - Raising θ by CV lowers but does not remove this: 0.16, 0.24 and 0.21.
  - CAS-NLI θ-CV preserve stays at 0.04–0.07, the same range as BM25 and CAS-keyword. Vanilla RAG gives 0.
- (b) Abstentions. Every filter increases abstention, because filtering sometimes removes the passage that holds the answer.
  - Vanilla RAG: 0.12–0.20.
  - CAS-NLI θ-CV preserve: 0.20–0.29.
  - CAS-NLI: 0.40–0.54.
- (c) WikiContradict conflict flagging (n = 50).
  - Vanilla RAG gives 0.48, 0.70 and 0.80. CAS-NLI θ-CV preserve gives 0.32, 0.34 and 0.58.
  - The labelled conflict notes do **not** raise flagging above simply showing both passages. Every significant difference is a drop vs Vanilla RAG:
    - Gemma, θ-CV preserve: −0.36 (Holm p = 0.021).
    - Gemma, CAS-keyword: −0.28.
    - Llama 3.1 8B, CAS-NLI: −0.52.
    - Llama 3.1 8B, CAS-NLI θ-CV: −0.32 (Holm p = 0.040).

**Fig. 10. Answer accuracy vs generation cost** (HotpotQA and MuSiQue pooled, n = 65 per point).
- Latency is LLM generation only; it excludes the NLI step in Fig. 5.
- Filtering cuts the mean prompt from about 1,700–1,770 tokens to 270–560, and generation time from 13–30 s to 1.6–8 s.
- CAS-NLI θ-CV preserve uses about 28% of Vanilla's prompt. The accuracy cost:

  | Model | θ-CV preserve | Vanilla RAG | Gap |
  |---|---|---|---|
  | Gemma 3 4B | 0.446 | 0.477 | −0.03 |
  | Llama 3.2 3B | 0.431 | 0.538 | −0.11 |
  | Llama 3.1 8B | 0.508 | 0.600 | −0.09 |

**Fig. 11. Choosing the contradiction threshold θ by 5-fold cross-validation** (HotpotQA + MuSiQue, n = 65).
- (a) Mean evidence F1 over the θ grid, all questions, descriptive.
  - With the penalty, F1 rises monotonically from 0.320 (θ = 0.5) to 0.416 (θ = 0.99). In other words, the fewer contradictions the graph acts on, the better it does.
  - The conflict-preserving mode is flat (0.577–0.579).
  - All five folds chose θ = 0.99, the edge of the grid, for both modes.
- (b) F1 on the four training folds (bar) vs the held-out fold (marker), per fold. The held-out F1 of the penalty varies between 0.32 and 0.60 across folds.

## Summary of evidence

| Claim | Supported? |
|---|---|
| A real NLI model detects contradictions that the keyword verifier misses | **Yes**: F1 0.652 vs 0 (Fig. 4a) |
| Penalising contradicting passages improves evidence selection | **No**: it removes 37% of gold evidence and sometimes all of it (Figs. 3, 7, 9a). A held-out θ reduces the damage but does not remove it, and CV pushes θ to its maximum (Fig. 11) |
| Conflict-preserving CAS-NLI selects better evidence than the original CAS and Vanilla RAG | **Yes**, with a held-out θ (Fig. 2) |
| …than plain BM25 top-3 | Higher precision only; F1 not significant (Fig. 2) |
| …than CAS without the conflict graph | **No** difference (Fig. 2): the graph is not what helps |
| CAS improves LLM answer accuracy | **No** significant improvement with 3B, 4B or 8B models (Fig. 8) |
| CAS reduces prompt size and generation latency | **Yes**, about 3.5× shorter prompts (Fig. 10), but the NLI step itself costs 23–84 s per query on CPU (Fig. 5) |
| CAS improves conflict surfacing | **No**: showing both passages (Vanilla) does as well or significantly better (Fig. 9c) |

## Limitations

- **Small samples.** n = 50 (HotpotQA), 15 (MuSiQue), 50 (WikiContradict), 50 (RAGTruth), so CIs are wide.
- **Model size.** The largest LLM tested is an 8B model on CPU. Larger models (≥ 70B), or API models, may use conflict notes differently.
- **θ at the grid edge.** CV chose θ = 0.99 every time, so the optimum may lie above the grid. That would mean an even weaker contradiction signal, which is consistent with the penalty being harmful.
- **Automatic answer metrics.** These are string match and keyword cues. A human-annotated sample of `llm_generations.jsonl` is needed to validate them.
- **No source metadata.** Provenance is a constant prior (0.5), because the benchmarks carry no source reliability. CAS's trust component is therefore only partly exercised.
- **Self-RAG and CRAG are not implemented.** The comparison numbers previously in `results_cas.py` and the dashboards were not measured, and must not be cited.

## Audit of figures previously in the repository

`cas_graphs_without_legend/`, `cas_graphs_without_legend.zip` and `extracted_graphs/*.png` were generated by `results_cas.py` from hardcoded values. They include two datasets not in `data/` (RGB, MultiDoc2Dial) and a baseline CPI that contradicts `compute_cpi` (77.5 vs 75.53). **Do not submit them.** `extracted_graphs/mergered.py` cannot run (missing imports and definitions).
