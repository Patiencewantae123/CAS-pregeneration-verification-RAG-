# Journal figures: CAS pre-generation verification

Every figure here is drawn from `evaluation/results/results.json`, which `evaluation/run_evaluation.py` produces by running the CAS code in `python/` on the datasets in `data/`. No value is typed in by hand. `make_all.py` rebuilds all seven figures and checks the plotted CSV values against `results.json` (last run: **PASS**).

## Reproduce

```bash
py evaluation/run_evaluation.py     # ~5 s on CPU; writes evaluation/results/*
py journal_figures/make_all.py      # writes figNN_*.png (600 dpi), .pdf (vector), data/figNN_*.csv
```

Requirements: Python 3.14, numpy 2.5, pandas 3.0, matplotlib 3.11. Seed 42; 2,000 bootstrap resamples; 10,000 sign-flip permutations.

Timing numbers in Fig. 5 change slightly from run to run. Every other number is deterministic.

## Figure → data mapping

| Figure | Script | Plotted data | Source in `results.json` |
|---|---|---|---|
| Fig. 1 | `fig01_precision_recall_tradeoff.py` | `data/fig01_precision_recall_tradeoff.csv` | `evidence_selection` |
| Fig. 2 | `fig02_paired_differences.py` | `data/fig02_paired_differences.csv` | `paired_tests` |
| Fig. 3 | `fig03_conflict_graph_ablation.py` | `data/fig03_conflict_graph_ablation.csv` | `graph_diagnostics` |
| Fig. 4 | `fig04_conflict_hallucination_detection.py` | `data/fig04_conflict_hallucination_detection.csv` | `contradiction_by_group`, `hallucination` |
| Fig. 5 | `fig05_performance_cost.py` | `data/fig05_performance_cost.csv` | `timing`, `evidence_selection` |
| Fig. 6 | `fig06_threshold_sensitivity.py` | `data/fig06_threshold_sensitivity.csv` | `threshold_sweep` |
| Fig. 7 | `fig07_error_analysis.py` | `data/fig07_error_analysis.csv` | `error_analysis` |

## Captions

**Fig. 1. Evidence-selection precision and recall.** Each point is the mean over questions, with 95% bootstrap confidence intervals on both axes. Dotted curves are iso-F1 contours.
- (a) HotpotQA, n = 50 questions with 10 candidate paragraphs each.
- (b) MuSiQue, n = 15 questions with 20 candidates and at least one gold paragraph.

On HotpotQA, CAS (full) reaches P = 0.589 and R = 0.660 (F1 = 0.580). Without the conflict graph it reaches P = 0.752 and R = 0.630 (F1 = 0.657).

**Fig. 2. Paired per-question differences between CAS (full) and each comparator.** Error bars are 95% bootstrap CIs. Filled markers show Holm-adjusted p < 0.05 from a two-sided paired sign-flip permutation test across 18 comparisons.
- **CAS vs Vanilla RAG:** CAS improves F1 by +0.247 [0.182, 0.313] on HotpotQA and +0.189 [0.111, 0.276] on MuSiQue.
- **CAS vs BM25 top-3:** the F1 difference is not significant on either dataset (−0.012 and −0.025, Holm p = 1.00).
- **Removing the conflict graph *raises* HotpotQA F1:** CAS (full) scores −0.076 [−0.123, −0.033] lower (Holm p = 0.017).

**Fig. 3. What the conflict graph does.**
- (a) Share of passage pairs that the CAS `NLIVerifier` labels Entailment (E), Contradiction (C) or Neutral (N). Not one of the 5,100 pairs is labelled Contradiction.
- (b) Mean agreement credit given to gold versus distractor passages. On HotpotQA the credit is nearly equal (0.217 vs 0.214). On MuSiQue, distractors receive twice as much as gold passages (0.179 vs 0.088).

**Fig. 4. Conflict and hallucination detection.**
- (a) Fraction of passage pairs flagged as contradictory, with Wilson 95% CIs. Pairs marked (+) are known contradictions (WikiContradict). Pairs marked (−) are consistent gold-evidence pairs (HotpotQA). The CAS verifier flags 0 of 120 pairs. A negation-cue baseline flags 14.3% of explicit and 13.6% of implicit contradictions, and 8.0% of consistent pairs.
- (b) ROC curves for detecting hallucinated RAGTruth summaries (12 of 50). An unsupported-token-rate score reaches AUROC 0.81 [0.69, 0.92]. The CAS verifier labels every output Neutral, so its AUROC is 0.50.

**Fig. 5. Accuracy versus latency.** Evidence F1 with 95% CIs, plotted against selection latency per query (mean ± SD over 20 repeats, single CPU thread, log scale). Building the conflict graph is quadratic in the number of candidates. It raises latency by about 4.5× on HotpotQA (1.24 vs 0.28 ms; 10 candidates) and about 8.6× on MuSiQue (4.17 vs 0.48 ms; 20 candidates), without improving F1.

**Fig. 6. Sensitivity to the trust threshold τ.** HotpotQA and MuSiQue pooled, n = 65.
- F1 rises from 0.532 at the default τ = 0.45 to 0.594 at τ = 0.65.
- Above τ = 0.675, F1 collapses to 0.167. 0.675 is the highest trust score a passage can reach without agreement credit (0.35·relevance + 0.25·0.5 provenance prior + 0.20·rank confidence).
- τ was swept on the evaluation data itself, so the peak describes behaviour and is not a tuned result.

**Fig. 7. Where CAS (full) decisions at τ = 0.45 go wrong.**
- (a) Gold passages. Of the 34 HotpotQA gold passages that are lost, 25 rank outside BM25's top 3, so the cause is weak lexical relevance, not conflict handling.
- (b) Distractor passages. Of the 72 HotpotQA distractors that CAS retains, 37 get through **only** because of agreement credit from the conflict graph. On MuSiQue the figure is 16 of 54.

## Audit of figures already in the repository

| Location | Verdict | Reason |
|---|---|---|
| `cas_graphs_without_legend/*.png`, `cas_graphs_without_legend.zip`, `extracted_graphs/*.png` | **Do not submit** | Generated by `results_cas.py` from hardcoded numbers that no code in the repository produces. Two of the plotted datasets (RGB, MultiDoc2Dial) are not in `data/`. Self-RAG and CRAG are not implemented. With no legend, the four series cannot be told apart. Lines join categorical datasets, which implies a trend that doesn't exist. The plotted baseline CPI (77.5) disagrees with `CASEngine.compute_cpi` applied to the same table (75.53). |
| `extracted_graphs/mergered.py` | Broken | Imports, `DATASETS`, `METHODS`, `METRICS` and `OUTPUT_DIR` are undefined, so it cannot run. |
| `evaluation/figures/*` | Superseded | Earlier 300-dpi drafts. Replaced by this folder. |

The same hardcoded table also appears in `python/app.py` (Benchmark tab), `react-dashboard/src/models/casData.js` and `python/main.py`. Do not cite it as a measured result.

## Requested figures that cannot be produced from the available data

| Requested figure | Missing data | Experiment needed |
|---|---|---|
| Comparison with Self-RAG and CRAG | Neither method is implemented | Implement or run reference code for both methods on the same candidate pools |
| Hallucination and contradiction **rates of generated answers**; CPI | No generation step exists: the pipeline stops at `C_verified` | Give the LLM each method's selected context, then score the answers (e.g. RAGTruth-style annotation, an NLI judge, or exact match against gold answers) |
| Comparison across LLMs | No model weights or API access configured | Choose the models and provide access; the generation stage plugs in after evidence selection |
| Variability across independent runs | The pipeline is deterministic, so repeated runs are identical | Variability is reported with bootstrap CIs over questions instead; resampling the candidate pools would add a second source |
| Memory usage | Not measured | Add `tracemalloc` around selection (negligible for this pure-Python heuristic) |
