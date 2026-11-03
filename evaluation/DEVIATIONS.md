# Deviations from the pre-registered analyses

## 2026-11-03: scoring corrections after a code review

A review of the evaluation code, made after studies 1–3 had been run, found implementation errors in how answers were scored. The pre-registered *definitions* are unchanged; the code did not implement them as written. For example, study 3 defines accuracy as "the gold answer appears in the first line of the answer", but the code also counted "no" as appearing inside "unknown".

**How the correction was made:**
- Every study was re-scored from the same cached model outputs. No model was re-run and no selection changed.
- The as-run result files are archived unchanged in `results/as_run_2026-11-03/`, and they are also in git history.
- `compare_as_run.py` writes the full before/after comparison to `results/deviation_impact.json`.

### What was corrected

| # | Error | Fix | Affects |
|---|---|---|---|
| 1 | Accuracy used a raw substring match, so a gold answer "no" was found inside "unknown" and short answers matched inside longer words | Whole-word match after normalisation; an abstention is never counted as correct | Accuracy in all studies |
| 2 | In study 2, a fake answer that contains the real answer ("Java" → "Invasion of Java"), or the reverse, scored as both correct and adopted | The longer answer is removed before searching for the shorter one | Accuracy and adoption, studies 1–2 |
| 3 | Poison adoption, answer-in-context, answer coverage and the Monte Carlo answer-availability endpoint used raw substring matches | Whole-word match | Adoption (studies 1–2); Table 3; Fig. 15 |
| 4 | The conflict flag counted any citation ("Passage [1] states…") or "however" as flagging a conflict | Explicit conflict language only | WikiContradict flagging (Fig. 9c, Table 9), exploratory |
| 5 | The negation-cue baseline never saw contracted negations ("isn't") | Contractions recognised | Fig. 4a, Table 6 (baseline only) |
| 6 | NLI latency SDs came from cache-hit repeats, so a ~23 s model step was reported as exact to ±0.03 ms | No run-to-run SD for NLI methods; their spread across questions is reported instead | Fig. 5 |

### Impact on the pre-registered decisions

**No pre-registered decision changed in any study.**

| Study | Decisions as run | Decisions corrected |
|---|---|---|
| 1 (HotpotQA-Poison) | 5 × "not supported" | 5 × "not supported" |
| 2 (HotpotQA-Poison-2) | 5 × "not supported" | 5 × "not supported" |
| 3 (HotpotQA-500) | H1, H2, H3 not supported | H1, H2, H3 not supported |

**Confirmatory tests whose numbers or significance moved:**

| Test | As run | Corrected |
|---|---|---|
| Study 1: Llama 3.2, condition A, accuracy, preserve − Vanilla | −0.115, Holm p = 0.004 | −0.110, Holm p = 0.012 (still significant) |
| Study 2: Llama 3.2, condition B, adoption, preserve − source filter | +0.195, Holm p = 0.002 | +0.190, Holm p = 0.002 |
| Study 3, H2: Gemma 3, accuracy, preserve − Vanilla | −0.052, Holm p = 0.112 | **−0.068, Holm p = 0.010 (now significant)** |
| Study 3, H3: Llama 3.2, CI lower bound vs −0.05 margin | −0.080 (fails) | −0.078 (fails) |
| Study 3, H3: Gemma 3, CI lower bound vs −0.05 margin | −0.092 (fails) | −0.108 (fails) |

Every other confirmatory test in studies 1–3 kept its significance status. Each moved by at most 0.005 in effect size; most did not move at all.

**The one change in significance is against CAS.** Before the fix, "unknown" was sometimes scored as correct, which favoured the methods that abstain more. In study 3, conflict-preserving CAS abstains far more often than Vanilla RAG: 24.0% vs 8.2% for Gemma 3, and 15.8% vs 3.4% for Llama 3.2.

**Exploratory results that changed:**
- **Conflict flagging on WikiContradict (Fig. 9c).** Four significant drops in flagging for CAS variants were artefacts of the citation rule (error 4) and disappear. After correction, no flagging difference is significant, and the conclusion "conflict notes do not raise flagging" is unchanged.
- **Clean-benchmark accuracy (Figs. 8 and 10).** It fell for every method, because correct-looking abstentions are no longer counted. No CAS variant becomes significantly better or worse than before. The CAS-NLI (argmax) losses for Llama 3.1 8B become larger:
  - vs Vanilla RAG: −0.38 (was −0.32);
  - vs BM25 top-3: −0.36 (was −0.32).
- **Preserve vs penalty under poisoning (study 1, exploratory family).** The difference is now +0.060 to +0.065, significant in three of four cases (Holm p = 0.031–0.044). Before, it was +0.065 to +0.070, Holm p ≈ 0.03.
- **Monte Carlo (Fig. 15).** Answer availability moved by at most 0.01. The silent-exposure results, the primary endpoint, are unchanged.
- **Detector probe (Fig. 14).** Unchanged.

## Other code fixes on the same day (no effect on any reported number)

- The NLI and LLM caches skip lines cut short by an interrupted write, instead of crashing.
- θ-CV methods raise an error if `cv_thresholds.json` is missing, instead of silently using θ = 0.5.
- `boot_auroc` refuses a single-class label set instead of looping forever.
- Threshold views of the NLI verifier share one model load.
- The detector probe and Fig. 14 handle the case where no setting meets the false-alarm rule.
- `human_eval/make_sheet.py` never overwrites a sheet that already holds labels.
- `run_llm_generation.py` defaults to all three models, so a re-run no longer drops a model from the summary.
- Holm correction is one shared function (`run_evaluation.holm`), where there used to be three copies.
- `journal_figures/make_all.py` now also checks Figs. 12–15 against their result files, and exits with an error when a check fails.
