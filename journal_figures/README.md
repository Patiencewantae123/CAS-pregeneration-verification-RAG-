# Journal figures: CAS pre-generation verification

Every number in this folder is measured. The figures are drawn from the files in `evaluation/results/` listed in the mapping table below, which the evaluation scripts produce by running the CAS code in `python/` on the datasets in `data/`. Nothing is typed in by hand. `make_all.py` rebuilds all sixteen figures and `tables.md`, then checks the plotted values against the results files and exits with an error if any differ (last run: **PASS, 16 figures**).

## Reproduce

```bash
py evaluation/run_evaluation.py         # ~45 min on CPU (DeBERTa NLI); predictions cached in nli_cache.jsonl
py evaluation/run_llm_generation.py --models llama3.2:latest gemma3:latest llama3.1:8b
                                        # Ollama, CPU only; ~3.5 h for the 8B model; responses cached
py evaluation/run_clean500.py           # study 3: 500 questions x 6 methods x 2 LLMs; ~6 h on CPU, responses cached
py evaluation/run_montecarlo.py         # study 4: 3,064,800 simulated selections in ~1 min; no model inference
py evaluation/compare_as_run.py         # as-run vs corrected scores (DEVIATIONS.md)
py evaluation/run_discussion_analyses.py  # exploratory study 3 analyses for manuscript Section 6 (Tables 4-5)
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
6. **Scoring corrections (2026-11-03), after a code review.** Every study was re-scored from the same cached model outputs, with no model re-run:
   - answers now match the gold answer as whole words ("no" is no longer found inside "unknown");
   - an abstention is never counted as correct;
   - the conflict flag needs explicit conflict language;
   - fake answers that contain the real answer, or the other way round, are scored separately.

   No pre-registered decision changed. The as-run files are archived in `evaluation/results/as_run_2026-11-03/`. Details and the impact on every test are in `evaluation/DEVIATIONS.md`.

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
| 12 | `fig12_poison_study1.py` (uses `_poison.py`) | `data/fig12_*.csv` | `poison_summary.json` → `descriptive` |
| 13 | `fig13_poison_study2.py` (uses `_poison.py`) | `data/fig13_*.csv` | `poison2_summary.json` → `descriptive` |
| 14 | `fig14_detector_tradeoff.py` | `data/fig14_*.csv` | `detector_probe.json` (from `evaluation/run_detector_probe.py`) |
| 15 | `fig15_montecarlo.py` | `data/fig15_*.csv` | `montecarlo_summary.json`, `montecarlo_e1/e2/e3.csv` (from `evaluation/run_montecarlo.py`) |
| 16 | `fig16_clean500.py` | `data/fig16_*.csv` | `clean500_summary.json` (from `evaluation/run_clean500.py`) |

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
- Log scale. Methods without NLI: mean ± SD over 20 repeats.
- NLI methods include the CPU model time, which was measured once per passage pair when it was first computed. They therefore have no run-to-run SD and are shown without error bars. Their spread across questions is in `results.json` → `timing` → `nli_model_ms_sd_across_queries`.
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
- On HotpotQA, Vanilla RAG is the most accurate method for every model: 0.56 (Llama 3.2 3B), 0.52 (Gemma 3 4B) and 0.68 (Llama 3.1 8B).
- The larger model does not change the ranking. With Llama 3.1 8B:
  - BM25 top-3 reaches 0.66.
  - CAS-keyword reaches 0.58, and CAS-NLI θ-CV preserve 0.56.
  - CAS-NLI θ-CV (penalty) reaches 0.46, and CAS-NLI reaches 0.30.
- After Holm correction over 210 tests, **no CAS variant is significantly better than any baseline for any model**. The significant QA differences are all losses for CAS-NLI with argmax labels:
  - Gemma, HotpotQA F1 vs Vanilla: −0.283.
  - Llama 3.1 8B, HotpotQA vs Vanilla: accuracy −0.38, F1 −0.366.
  - Llama 3.1 8B, HotpotQA vs BM25: accuracy −0.36, F1 −0.333.
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
  - Flagging is rare for every method:
    - Vanilla RAG: 0.22 (Llama 3.2), 0.32 (Gemma), 0.28 (Llama 3.1 8B).
    - CAS-NLI θ-CV preserve: 0.14, 0.22 and 0.16.
  - The labelled conflict notes do **not** raise flagging above simply showing both passages. No flagging difference is significant after Holm correction.
  - The only significant conflict-handling difference is a drop: with Llama 3.2, CAS-NLI with argmax labels gives both answers less often than Vanilla RAG (−0.32, Holm p = 0.021).
  - These numbers use the corrected flag rule. The first version counted any citation such as "Passage [1] states…" as a flag; see `evaluation/DEVIATIONS.md`.

**Fig. 10. Answer accuracy vs generation cost** (HotpotQA and MuSiQue pooled, n = 65 per point).
- Latency is LLM generation only; it excludes the NLI step in Fig. 5.
- Filtering cuts the mean prompt from about 1,700–1,770 tokens to 270–560, and generation time from 13–30 s to 1.6–8 s.
- CAS-NLI θ-CV preserve uses about 28% of Vanilla's prompt. The accuracy cost:

  | Model | θ-CV preserve | Vanilla RAG | Gap |
  |---|---|---|---|
  | Gemma 3 4B | 0.385 | 0.431 | −0.05 |
  | Llama 3.2 3B | 0.400 | 0.492 | −0.09 |
  | Llama 3.1 8B | 0.462 | 0.585 | −0.12 |

**Fig. 11. Choosing the contradiction threshold θ by 5-fold cross-validation** (HotpotQA + MuSiQue, n = 65).
- (a) Mean evidence F1 over the θ grid, all questions, descriptive.
  - With the penalty, F1 rises monotonically from 0.320 (θ = 0.5) to 0.416 (θ = 0.99). In other words, the fewer contradictions the graph acts on, the better it does.
  - The conflict-preserving mode is flat (0.577–0.579).
  - All five folds chose θ = 0.99, the edge of the grid, for both modes.
- (b) F1 on the four training folds (bar) vs the held-out fold (marker), per fold. The held-out F1 of the penalty varies between 0.32 and 0.60 across folds.

**Fig. 12. Pre-registered study 1: CAS under poisoned retrieval** (HotpotQA-Poison, 200 new dev questions; `evaluation/PREREGISTRATION_poison.md`).
- **Setup.** Each question gets one or two copies of its gold passage with the answer replaced by another question's answer. Condition B adds noisy source-reliability scores, which every method can use. Nothing was tuned (τ = 0.45, θ = 0.99).
- **(a, b) Accuracy.** Vanilla RAG is the most accurate (Llama 0.505, Gemma 0.390). CAS-NLI preserve is significantly less accurate than Vanilla for Llama in condition A (−0.110 [−0.165, −0.055], Holm p = 0.012). Among the methods that filter the context, it is the strongest after CAS-keyword:
  - vs BM25 top-3: +0.045 to +0.085.
  - vs Provenance + BM25 top-3: +0.020 to +0.090.
  - Some of these differences (condition B) are significant before Holm correction; none is after it.
  - vs CAS-NLI penalty: +0.060 to +0.065, significant in three of four cases (exploratory family, Holm p = 0.031–0.044).
- **(c, d) Poison adoption.** The models seldom gave the fake answer even with no filtering (Vanilla RAG: 3.5% and 6.5%), because many substitutes were implausible. This floor left no room to show a reduction. In condition B the source filter did best: for Gemma it cut adoption to 1.0%, against 8.0% for CAS-NLI preserve (Holm p = 0.030).
- **Pre-registered decisions:** all five are "not supported".

**Fig. 13. Pre-registered study 2: a stronger attack** (HotpotQA-Poison-2, 200 further dev questions; `evaluation/PREREGISTRATION_poison2.md`, including its one amendment, made before any method ran).
- **Setup.** Three type-matched poison passages per question, which agree with each other and outnumber the true passage, following the PoisonedRAG black-box design. Conditions, methods and tests are the same as in study 1.
- **The attack worked.** With no filtering, the models gave the fake answer in 28.0% (Llama) and 28.5% (Gemma) of answers, and accuracy fell to 0.230 and 0.220.
- **(a, c) Condition A, no source metadata.** CAS-NLI preserve does not differ from Vanilla RAG or BM25 top-3 on either endpoint (all Holm p = 1.0). The poison reaches its context for every question.
- **(b, d) Condition B, noisy source metadata.**
  - Provenance + BM25 top-3 cuts adoption to 3.5% (Llama) and 4.0% (Gemma), and its accuracy is not significantly different from CAS-NLI preserve.
  - CAS-NLI preserve stays at 22.5% and 28.0%, significantly worse (+0.190 and +0.240, both Holm p = 0.002).
  - The poison still reaches the CAS-NLI preserve context for every question. The trust score does not remove a highly relevant passage on provenance alone (CAS w/o graph: 99.5%), and conflict preservation adds contradicted passages back.
- **CAS-NLI penalty** removes poison from about a quarter of contexts and gives the lowest CAS adoption (18.5–21.5%), with abstention up to 20.5–28.5% and no accuracy gain. Its adoption is significantly lower than CAS-NLI preserve in three of four cases (exploratory family).
- **Pre-registered decisions:** all five are "not supported".

**Fig. 14. Can the conflict detector be made accurate and cheap enough for day-to-day use?** This analysis is exploratory (`evaluation/run_detector_probe.py`).
- **Ground truth.**
  - The gold passage vs its poisoned copy is a true conflict.
  - Pairs of original HotpotQA passages are treated as non-conflicting, so a flag there is a false warning on a normal question.
  - 50 WikiContradict pairs give natural contradictions.
- **Configurations.** 110 detector settings, covering:
  - NLI model: DeBERTa-v3 or MiniLM.
  - What is compared: whole passages or the most question-relevant sentence.
  - θ ∈ {0.5, 0.9, 0.99}.
  - A lexical-overlap gate that skips unrelated pairs.
  - One or both directions.
  - Also answer-level detection with two extractive QA models.
- **Selection rule (fixed in advance):** on study 1, maximise conflicts caught subject to at most 10% false warnings. Study 2 is the held-out check.
- **(a) Detection vs false alarms.**
  - The current CAS detector (★) catches 14.5% of held-out conflicts and 14% of WikiContradict contradictions, but gives a false warning on 52.5% of normal questions.
  - No setting catches more than 14% of held-out conflicts while staying within 10% false warnings.
  - The rule-chosen settings (circled) catch 3.0% (MiniLM) and 7.0% (answer-level) of held-out conflicts.
- **(b) Detection vs CPU cost.**
  - CPU cost is not the obstacle. Sentence-level MiniLM with the overlap gate needs 0.04 s per query against 21 s for the current detector, about 500 times less, and some cheap settings catch up to 70% of conflicts.
  - But those settings warn on about a third of normal questions.
- **Conclusion.** The binding constraint is detection quality, not speed.

**Fig. 15. Pre-registered Monte Carlo study of the CAS selection layer (study 4; 3,064,800 question-level runs).** The script is `evaluation/run_montecarlo.py`; the plan is in `evaluation/PREREGISTRATION_montecarlo.md`.
- **What is simulated.**
  - Only which passages reach the prompt. No LLM is run.
  - BM25 scores and DeBERTa-v3 predictions are the real ones for the 400 questions of studies 1–2.
  - Before anything was simulated, the re-implementation reproduced all 4,400 stored study 1–2 selections exactly.
- **Primary endpoint: silent poison exposure.** A poisoned passage reaches the prompt without a conflict note.
- **The unit of inference is the question (n = 400).** The run count removes simulation noise; it is not sample size, so no p-values are computed over runs.
- **(a) Trust weights and τ (E1).**
  - Settings: 1,250 random ones (weights ~ Dirichlet(1, 1, 1, 1), τ ~ U(0.30, 0.60)) plus the default.
  - **Against Vanilla RAG:** CAS-NLI preserve has less silent exposure in 100% of settings (pooled mean difference −0.22 without metadata, −0.27 with noisy metadata). It keeps the answer passage less often in every setting, because Vanilla RAG keeps everything.
  - **Against the source filter (Provenance + BM25 top-3):** CAS has more silent exposure in 100% of settings (+0.50, 95% CI [+0.47, +0.52]). It keeps the answer passage more often in 92% of settings (+0.18 [+0.15, +0.20]).
  - **Against BM25 top-3:** the silent-exposure difference is not robust to the parameters (CAS is better in 36% of settings without metadata and 62% with noisy metadata). CAS keeps the answer passage more often (+0.19 and +0.23).
  - **Conclusion:** study 2's finding that the source filter beats CAS is not an artefact of the hand-set weights.
  - **Note:** at the low end of its range (5th percentile), CAS w/o graph keeps no passage at all, so its exposure is 0 there.
- **(b) Source-metadata quality (E2).**
  - Axes: the separation between trusted and untrusted scores (x) and the share of mislabelled sources (y). The boxed cell is condition B of studies 1–2.
  - The source filter reaches ≤ 5% silent exposure only when no source is mislabelled and the separation is ≥ 0.4. With 10% mislabelled sources it bottoms out at 12%.
  - CAS-NLI preserve barely responds to metadata quality (0.72–0.79 across the whole grid). It lets more poison through silently than the source filter in every cell. Provenance enters as one weight of 0.25, so even perfect metadata does not outweigh relevance and agreement.
- **(c, d) Detector quality (E3).**
  - Setup: a synthetic detector flags each true conflict with probability r (x axis) and adds false alarms on 0–50% of normal questions (line styles). The lines overlap, so false alarms barely change this endpoint.
  - ★ marks the current DeBERTa detector (r = 0.08 on these pairs). It sits below the curves because it also flags poison against passages other than its source (359 flags in 126 of 400 questions), which the synthetic detector never does.
  - **Without metadata (c):** CAS-NLI preserve matches BM25 top-3 from r = 0.3. That corresponds to a question-level recall of about 50%, about five times the current detector's 9.75%.
  - **With noisy metadata (d):** CAS-NLI preserve never reaches the source filter, even with a perfect detector (0.33 vs 0.23). CAS-NLI penalty crosses it between r = 0.8 and 0.9, but by then it keeps the answer passage in at most 25% of questions (e).
- **(e) The cost, at 10% false alarms.**
  - Penalty mode penalises both sides of a detected conflict. Going from no detection to a perfect detector cuts how often the answer passage is kept from 70–77% to 10–16%.
  - Preserve mode also drops, from 71–78% to 45–58%. Reason: the real model labels 470 of 845 gold–poison pairs (56%) as *agreement*, and only 67 (8%) as contradiction. Flagging a pair removes that agreement bonus from both passages.
  - The same agreement bonus explains why, with no contradiction flags at all, CAS-NLI lets more poison through silently than CAS w/o graph (0.84 vs 0.77 without metadata): a poisoned copy "agrees" with its source, and the trust score rewards it.
- **E4 (exploratory, real LLM outputs; not plotted; numbers in `montecarlo_summary.json`).**
  - When poison reached the CAS-NLI preserve prompt, a conflict note did not lower adoption.
    - Llama 3.2: 15.7% with a note (n = 70) vs 16.8% without (n = 594).
    - Gemma 3: 28.6% vs 22.2%.
  - For these small models, a better detector would not help unless the generator acts on the notes.

**Fig. 16. Pre-registered study 3: clean retrieval at n = 500** (500 further HotpotQA dev questions; `evaluation/PREREGISTRATION_clean500.md`; scores corrected as described in `evaluation/DEVIATIONS.md`).
- **Setup.**
  - 10 passages per question, with no poisoning and no tuning (τ = 0.45, θ = 0.99).
  - A cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) is added as a stronger baseline.
  - Answers come from Llama 3.2 3B and Gemma 3 4B. Error bars are 95% bootstrap CIs. Filled markers in (c, d) have Holm p < 0.05 over the 9 confirmatory tests.
- **(a, c) Evidence selection, H1.**
  - CAS-NLI preserve reaches F1 0.669, against 0.335 for Vanilla RAG, 0.640 for BM25 top-3 and 0.648 for the reranker.
  - It is significantly better than Vanilla RAG (+0.334 [+0.316, +0.353], Holm p < 0.001) and BM25 top-3 (+0.029 [+0.016, +0.043], Holm p = 0.002).
  - It is not significantly better than the reranker (+0.021 [−0.001, +0.043], Holm p = 0.40).
  - H1 required all three comparisons to be significant, so it is **not supported**.
  - Exploratory: CAS w/o graph selects slightly better evidence than CAS-NLI preserve (F1 0.680, difference −0.011, Holm p < 0.001), so the conflict graph does not help selection.
- **(b, d) Answer accuracy, H2 and H3.**
  - Vanilla RAG is the most accurate method (Llama 0.502, Gemma 0.456). CAS-NLI preserve reaches 0.464 and 0.388.
  - No comparison favours CAS. For Gemma, CAS-NLI preserve is significantly less accurate than Vanilla RAG (−0.068 [−0.108, −0.030], Holm p = 0.010).
  - Non-inferiority to Vanilla RAG (H3, margin −0.05) fails for both models: the CI lower bounds are −0.078 and −0.108.
  - H2 and H3 are **not supported**.
- **What does hold, exploratory.**
  - Conflict preservation beats the contradiction penalty: accuracy +0.058 (Llama) and +0.078 (Gemma), F1 +0.112, all Holm p < 0.001.
  - CAS-NLI preserve uses about 28% of Vanilla RAG's prompt (406 vs 1,433–1,462 tokens) and generates about 3.5× faster (2.6–3.6 s vs 10.3–12.8 s).
  - It abstains more often: 15.8% vs 3.4% (Llama) and 24.0% vs 8.2% (Gemma).
  - About half of the questions that Vanilla RAG answers correctly and CAS-NLI preserve does not are abstentions: 29 of 61 (Llama) and 37 of 70 (Gemma). In the other direction, CAS-NLI preserve answers correctly 42 (Llama) and 36 (Gemma) questions that Vanilla RAG gets wrong.

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
| CAS improves conflict surfacing | **No**: conflict notes do not raise flagging above simply showing both passages, and no flagging difference is significant (Fig. 9c) |
| CAS reduces adoption of poisoned passages (pre-registered study 1) | **Not supported**; adoption was near floor for every method (Fig. 12) |
| CAS reduces adoption under a stronger attack (pre-registered study 2) | **Not supported**: no difference from Vanilla RAG, and a simple source filter is significantly better (Fig. 13) |
| CAS selects better evidence at n = 500 (pre-registered study 3, H1) | **Not supported** as pre-registered. It is significantly better than Vanilla RAG (+0.334) and BM25 top-3 (+0.029), but not the reranker (+0.021, Holm p = 0.40) (Fig. 16) |
| CAS improves answers, or costs no accuracy, at n = 500 (study 3, H2/H3) | **Not supported**: no gain against any baseline; significantly less accurate than Vanilla RAG for Gemma (−0.068); non-inferiority fails for both models (Fig. 16) |
| The conflict check can run cheaply on CPU | **Yes**: 0.04 s per query, about 500× faster than now (Fig. 14b) |
| The conflict check is reliable enough for day-to-day use | **No**: at most 14% of conflicts caught at ≤ 10% false warnings; the current detector warns falsely on about half of normal questions (Fig. 14a) |
| Among context filters, conflict preservation beats contradiction penalties | **Yes**: +0.060 to +0.065 accuracy under poisoning, significant in three of four cases (Fig. 12; exploratory family, Holm p = 0.031–0.044), and on clean data (Fig. 2). At the selection level, penalties also discard the answer passage as detection improves (Fig. 15e) |
| CAS's selection-level results are an artefact of the hand-set weights and τ | **No**: the direction against Vanilla RAG and against the source filter holds in 100% of 1,250 random settings (Fig. 15a) |
| CAS uses source metadata as well as a simple source filter | **No**: at every metadata quality tested, the filter lets less poison through silently (Fig. 15b) |
| A better conflict detector would make CAS competitive | **Partly**: without metadata, CAS matches BM25 top-3 once it catches about half of conflicts, about 5× today's detector, whatever the false-alarm rate. With noisy metadata it never matches the source filter (Fig. 15c–d) |
| The agreement term in the trust score helps against poisoning | **No**: the NLI model calls 56% of gold–poison pairs "agreement", so the bonus raises poison exposure (Fig. 15, E3 at r = 0) |
| A conflict note stops small LLMs adopting poison | **No evidence**: 15.7–28.6% adoption with a note vs 16.8–22.2% without (exploratory, E4) |

## Limitations

- **Small samples.** n = 50 (HotpotQA), 15 (MuSiQue), 50 (WikiContradict), 50 (RAGTruth), so CIs are wide.
- **Model size.** The largest LLM tested is an 8B model on CPU. Larger models (≥ 70B), or API models, may use conflict notes differently.
- **θ at the grid edge.** CV chose θ = 0.99 every time, so the optimum may lie above the grid. That would mean an even weaker contradiction signal, which is consistent with the penalty being harmful.
- **Automatic answer metrics.** These are string match and keyword cues. A human-annotated sample of `llm_generations.jsonl` is needed to validate them.
- **No source metadata.** Provenance is a constant prior (0.5), because the benchmarks carry no source reliability. CAS's trust component is therefore only partly exercised. Studies 1–2 (condition B) and the Fig. 15b simulation add synthetic metadata, but no real source-reliability labels.
- **The Monte Carlo study simulates selection only.** Its endpoints are what reaches the prompt, not answers. Its detector and metadata generators are synthetic, and its conclusions hold only within those models.
- **Self-RAG and CRAG are not implemented.** The comparison numbers previously in `results_cas.py` and the dashboards were not measured, and must not be cited.

## Audit of figures previously in the repository

`cas_graphs_without_legend/`, `cas_graphs_without_legend.zip` and `extracted_graphs/*.png` were generated by `results_cas.py` from hardcoded values. They include two datasets not in `data/` (RGB, MultiDoc2Dial) and a baseline CPI that contradicts `compute_cpi` (77.5 vs 75.53). **Do not submit them.** `extracted_graphs/mergered.py` cannot run (missing imports and definitions).
