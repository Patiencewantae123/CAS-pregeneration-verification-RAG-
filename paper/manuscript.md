# When Does Conflict-Aware Evidence Verification Help Retrieval-Augmented Generation? Four Pre-Registered Studies Under Clean and Poisoned Retrieval

*Draft v0.1, 2026-11-03. Placeholders in **[brackets]** must be filled from result files that do not yet exist. No placeholder may be filled by hand: each names the file and field it comes from.*

---

## Abstract

**Background.** Retrieval-augmented generation (RAG) passes retrieved passages to a language model, including passages that are irrelevant, contradictory or deliberately poisoned. Conflict-aware synthesis (CAS) is a pre-generation verification layer that addresses this. It scores each passage for relevance, source provenance, retrieval rank and agreement with other passages. It uses a natural-language-inference (NLI) model to detect contradictions. It then either penalises contradicting passages or keeps both sides and labels the conflict.

**Methods.** We ran an exploratory comparison and four pre-registered studies, all on CPU only, with parameters that were never tuned on test data. Each pre-registered study had a hash-stamped analysis plan, written before its data were generated.
- **Exploratory, clean retrieval:** HotpotQA, MuSiQue, WikiContradict and RAGTruth, with three local LLMs (3–8B parameters).
- **Studies 1 and 2:** two pre-registered poisoning studies on 400 new HotpotQA questions. One used entity substitution; the other used a stronger, PoisonedRAG-style attack.
- **Study 3:** a pre-registered 500-question replication on clean retrieval, with a cross-encoder reranker baseline.
- **Study 4:** a pre-registered Monte Carlo study of the selection layer (3,064,800 runs), which tested sensitivity to the trust weights, source-metadata quality and detector quality.

**Results.**
- On clean retrieval, conflict-preserving CAS selected more precise evidence than Vanilla RAG (F1 +0.299, Holm p = 0.015). It cut prompts to about 28% of Vanilla's length. It did not improve answer accuracy for any model.
- None of the ten pre-registered decisions in the poisoning studies supported CAS. Under the stronger attack, a plain source filter cut poison adoption to 3.5–4.0%; conflict-preserving CAS left it at 23–28% (Holm p = 0.002).
- **[S3: one sentence on H1–H3 from `evaluation/results/clean500_summary.json` → `decisions`.]**
- The Monte Carlo study shows that these outcomes do not depend on the hand-set weights: the comparisons with Vanilla RAG and with the source filter went the same way in 100% of 1,250 random settings. It also identifies two causes:
  - The NLI model labels 56% of poisoned copies as *agreeing* with their source, so CAS's agreement score rewards the poison.
  - Without source metadata, the detector would need about five times its current recall before CAS matches a BM25 baseline.

**Conclusion.** Conflict-preserving verification is a better design than contradiction penalties. But with current NLI detectors, trust weighting does not protect RAG against poisoning; a hard source filter does. We give the detector and metadata conditions under which conflict-aware verification could pay off.

**Keywords:** retrieval-augmented generation; knowledge conflicts; corpus poisoning; natural language inference; evidence selection; pre-registration; negative results.

---

## 1. Introduction

Retrieval-augmented generation (RAG) [1], [2] grounds a language model's answer in passages retrieved at query time. The retrieved set is rarely clean. It may contain:
- passages that are irrelevant to the question, which distract the model [3], [4];
- passages that contradict each other [5]–[8];
- passages planted by an adversary to force a chosen answer [9].

LLMs handle such contexts inconsistently. Depending on how coherent and how frequent the conflicting evidence is, they may follow it or ignore it [7].

One family of defences acts **before generation**: it decides which passages reach the prompt and how they are presented. Conflict-aware synthesis (CAS), the system studied here, is such a defence. CAS does four things:
1. It scores each retrieved passage by relevance, source provenance, retrieval rank and agreement with the other passages.
2. It builds a pairwise conflict graph with an NLI model.
3. It drops passages whose trust falls below a threshold.
4. It handles detected contradictions in one of two ways: it penalises them, or it preserves both sides and attaches a conflict note for the generator.

The design is intuitive, and several published systems share parts of it [10]–[13]. Whether it works has rarely been tested under conditions that allow a negative answer.

This paper reports such a test. We make four contributions:

1. **A reproducible CPU-only evaluation of CAS** against Vanilla RAG, BM25, a cross-encoder reranker and a source filter. It covers evidence selection, answers from three local LLMs, conflict surfacing and cost. All code, prompts, cached model outputs and figure data are released.
2. **Three pre-registered confirmatory studies.**
   - Two use poisoned retrieval of increasing strength: entity substitution [5], and a multi-passage, PoisonedRAG-style attack [9].
   - One is a 500-question clean-retrieval replication.
   - For each, the hypotheses, tests, multiplicity correction and decision rules were written down before the data existed, and hash-stamped in the outputs. Section 4.5 states what the version history can and cannot confirm. All results are reported.
3. **A pre-registered Monte Carlo study of 3,064,800 runs of the selection layer.** It separates the effect of CAS's hand-set parameters from the effects of source-metadata quality and detector quality. It re-uses the real retrieval scores and NLI predictions.
4. **A diagnosis of why CAS fails against poisoning, with design guidance.**
   - The NLI model treats a poisoned copy of a passage as *entailed* by its source, so CAS's agreement bonus rewards the poison.
   - Provenance, entered as one weight among four, cannot outweigh relevance.
   - Contradiction penalties discard the true passage together with the false one.

The headline result is negative. On clean retrieval, CAS improves evidence precision and shortens prompts. It does not improve answers, and it does not protect against poisoning; a simple source filter does. We think this is worth publishing because the failure is specific and measurable, and because it is easy to fix in part. Section 6 states what a detector would have to achieve for conflict-aware verification to pay off.

## 2. Related Work

**Retrieval-augmented generation.** RAG [1] and its many variants [2] condition generation on retrieved text. Later systems add self-critique of retrieved passages (Self-RAG [10]) or a lightweight retrieval evaluator that triggers corrective actions (CRAG [11]). These methods need trained components, and they are not evaluated here; Section 7 discusses this. Long and noisy contexts degrade LLM answers [3], [4], which motivates filtering before generation.

**Knowledge conflicts.** Longpre et al. [5] built entity-substitution conflicts between retrieved context and model memory. Chen et al. [6] studied conflicts among multiple retrieved sources. Xie et al. [7] showed that LLMs accept coherent counter-memory evidence. Xu et al. [8] survey the area. WikiContradict [14] collects real contradictions from Wikipedia, and RAGTruth [15] annotates hallucinations in RAG outputs. We use both.

**Corpus poisoning.** PoisonedRAG [9] showed that a few injected passages, written to be retrieved and to state a target answer, can control RAG output. Certified and aggregation-based defences have been proposed in response [12], [13]. Our study 2 follows the black-box PoisonedRAG design: several mutually consistent poison passages per question, which outnumber the true passage.

**NLI as a verifier.** Cross-encoder NLI models such as DeBERTa-v3 [16], trained on MultiNLI [17] and similar corpora, are widely used to check consistency between texts. Their behaviour on near-duplicate passages that differ in a single entity is central to our results (Section 6.1).

**Pre-registration.** Pre-registration fixes hypotheses and analyses before data collection, so that confirmatory and exploratory results can be told apart [18]. It is still rare in NLP evaluation. We use it to make negative results interpretable.

## 3. The CAS Framework

### 3.1 Pipeline

For a question *q* with retrieved passages *d₁…dₙ*, CAS proceeds in three steps.

**Step 1: document features.** Each passage gets three features:
- relevance *r*: BM25 [19] divided by the pool maximum;
- retrieval confidence *c* = 1 − rank/*n*;
- provenance *p* ∈ [0, 1]: a source-reliability score, set to 0.5 when no metadata exist.

**Step 2: conflict graph.** For every pair of passages, an NLI model labels the relation as entailment, contradiction or neutral.
- The labels come from `cross-encoder/nli-deberta-v3-base`, scored in both directions and averaged.
- A pair counts as a contradiction only if p(contradiction) ≥ θ.
- θ = 0.99 throughout. It was chosen by 5-fold cross-validation on the clean benchmark (Section 4.5) and never re-tuned.

**Step 3: trust score.** Each passage receives

  *trust* = 0.35·*r* + 0.25·*p* + 0.20·*c* + 0.20·min(1, *agree*),

where *agree* is the summed confidence of the passage's entailment edges.

### 3.2 Synthesis modes

- **Penalty mode** (the original design) subtracts 0.5 × the summed contradiction confidence. It keeps a passage if the result is ≥ τ = 0.45.
- **Conflict-preserving mode** keeps passages with *trust* ≥ τ and applies no penalty. If a kept passage contradicts a dropped passage of relevance ≥ 0.5, the dropped passage is restored. Every contradiction among kept passages is attached to the prompt as a note: "passages [i] and [j] contradict each other".

### 3.3 Ablations

- **CAS w/o graph** uses the trust score without the NLI step.
- **CAS-keyword** uses the original keyword-overlap "NLI" heuristic.

The weights and τ are the defaults of the original implementation. They were not tuned in any study; study 4 tests sensitivity to them.

## 4. Experimental Design

### 4.1 Data

**Table 1. Datasets.**

| Set | Use | n | Candidates per question |
|---|---|---|---|
| HotpotQA (distractor) [20] | Clean evidence selection and QA (exploratory) | 50 | 10 |
| MuSiQue [21] | Clean evidence selection and QA (exploratory) | 15 | 20 |
| WikiContradict [14] | Contradiction detection; conflict surfacing | 50 pairs | 2 |
| RAGTruth [15] | Hallucination detection | 50 | — |
| HotpotQA-Poison (study 1) | Pre-registered poisoning | 200 | 11–12 |
| HotpotQA-Poison-2 (study 2) | Pre-registered poisoning | 200 | 13 |
| HotpotQA-500 (study 3) | Pre-registered clean replication | 500 | 10 |

The poisoning and replication samples are drawn from the HotpotQA dev set. They exclude every question used earlier, and the three samples do not overlap.

**Study 1 (entity substitution, after [5]).**
- Each question gets one copy (155 questions) or two copies (45) of a gold passage.
- In each copy, the answer is replaced by another question's answer.
- Questions with yes/no answers or answers longer than five words are excluded.

**Study 2 (PoisonedRAG-style, after [9]).** Each question receives three poison passages. All three state the same type-matched substitute: a shifted year or date, a scaled number, or a title-matched entity. The three forms are:
1. the substituted gold paragraph;
2. the question followed by the substituted paragraph;
3. the title followed by the substituted answer sentences.

The substitute rules were amended once (Amendment 1), before any method ran.

**Source metadata.** Each poisoning study runs in two conditions:
- **Condition A:** no metadata; every passage has provenance 0.5.
- **Condition B:** noisy metadata. Trusted sources score U(0.55, 0.95), untrusted sources U(0.05, 0.45), and each passage's trusted/untrusted status is flipped with probability 0.2.

### 4.2 Methods compared

- **Vanilla RAG:** keeps all passages.
- **BM25 top-3.**
- **Reranker top-3** (study 3 only): `cross-encoder/ms-marco-MiniLM-L-6-v2` [22], [23].
- **Provenance + BM25 top-3** (condition B): a hard source filter, provenance ≥ 0.5, then BM25 top-3.
- **CAS:** w/o graph, keyword, NLI penalty, and NLI conflict-preserving. Conflict-preserving CAS is the primary method in every pre-registered study.

### 4.3 Generation

- **Models:** Llama 3.2 3B and Gemma 3 4B in all studies, plus Llama 3.1 8B on the clean benchmark [24], [25].
- **Runtime:** Ollama on CPU, temperature 0, seed 42.
- **Prompt:** one fixed QA prompt that asks for a short answer, or "unknown" when the passages do not contain it.
- **Hardware:** an Intel CPU with 15 GB RAM and no GPU.

### 4.4 Metrics

- **Evidence precision, recall and F1:** measured against the supporting-fact paragraphs.
- **Accuracy:** the gold answer appears in the first line of the response after SQuAD normalisation [26].
- **Poison adoption:** the substitute answer appears in the first line.
- **Abstention:** the model says "unknown" or a similar phrase.
- **Poison in context:** a poison passage was selected.
- **Prompt tokens and latency.**
- **Monte Carlo endpoints (study 4):** silent poison exposure (poison in the prompt with no conflict note), any exposure, and answer passage retained.

The automatic accuracy, abstention and adoption labels are being validated against blind human labels on 150 stratified responses, with Cohen's κ [27] reported. **[HV: κ and 95% CI per label from `evaluation/human_eval/agreement.json`.]**

### 4.5 Statistical analysis

- **Paired tests:** two-sided sign-flip permutation tests (10,000 permutations) on per-question differences.
- **Confidence intervals:** 95% percentile bootstrap, 2,000 resamples [28].
- **Multiplicity:** Holm correction [29] within each pre-specified family.
- **The unit of analysis is always the question.**

**θ selection.** θ was chosen by 5-fold cross-validation on the 65 clean questions, with each question scored using the θ chosen on the other folds. All folds chose θ = 0.99, the edge of the grid. The value was then frozen for every later study.

**Pre-registration.**
- Each plan states that it was written before its benchmark was built or its first method was run:
  - study 1: 2026-10-29;
  - study 2: 2026-10-30;
  - studies 3 and 4: 2026-11-03.
- **What the git history independently confirms:**
  - Study 2: the plan was committed before its benchmark.
  - Study 3: the plan was committed before any output. The plan shares a commit with the 500-question sample, which a seeded script draws deterministically from the plan's own rules.
  - Studies 1 and 4: the plan first appears in the same commit as the benchmark (study 1) or the outputs (study 4). For these two, the timing rests on the plan's stated date.
- Each output file stores the SHA-256 of its plan. The stored hashes match the files:

| Study | Plan hash (prefix) |
|---|---|
| 1 | 47224db4 |
| 2 | a06a32f1 |
| 4 | 2038df84 |
| 3 | **[S3: `preregistration_sha256`]** |

- **Confirmatory families:**
  - studies 1–2: 20 tests each (2 models × 5 comparisons × 2 endpoints), five mechanical decisions;
  - study 3: 9 tests plus a non-inferiority margin of −0.05.
- **The plans were not deposited in a public registry.** We release the full git history.

### 4.6 Monte Carlo design (study 4)

Study 4 re-implements the selection layer in vectorised form. It uses the real BM25 scores and the cached DeBERTa predictions for the 400 questions of studies 1–2, so no model inference is run.

**Validation gate.** Before any simulation, the re-implementation had to reproduce every stored selection and conflict note of studies 1–2. It reproduced all 4,400 exactly.

**Experiments.**
- **E1:** 1,250 random weight vectors, Dirichlet(1, 1, 1, 1), and thresholds τ ~ U(0.30, 0.60), plus the default setting. Every setting ran in both conditions: 1,000,800 runs.
- **E2:** a 7 × 6 grid of metadata quality, varying the trusted–untrusted separation and the flip rate: 1,008,000 runs.
- **E3:** a synthetic detector that flags each true conflict with probability *r* and adds false alarms on 0–50% of normal questions: 1,056,000 runs.
- **E4 (exploratory):** whether conflict notes changed adoption in the real study 1–2 outputs.

Draws are averaged within each question before the bootstrap, and no p-values are computed over draws.

## 5. Results

### 5.1 Clean retrieval (exploratory)

**Evidence selection.**
- Conflict-preserving CAS with the cross-validated θ reached evidence F1 0.632 on HotpotQA (P 0.651, R 0.700).
- It improved on Vanilla RAG by +0.299 [0.233, 0.360] (Holm p = 0.015, over 150 tests) and on CAS-keyword by +0.080 (Holm p = 0.024).
- Against BM25 top-3 it was more precise (+0.158, Holm p = 0.015), but the F1 difference was not significant (+0.040 [−0.009, 0.087]).
- It was indistinguishable from CAS w/o graph (−0.018, Holm p = 1.0). **The conflict graph therefore added nothing to selection.**

**Contradiction penalties.**
- The penalty removed 37 of 100 HotpotQA gold passages.
- It left an empty context for 19 of 65 questions with argmax NLI labels.
- It was significantly worse than the same pipeline without the graph (F1 −0.203, Holm p = 0.024).

**Answers.**
- After Holm correction over 210 tests, no CAS variant was significantly better than any baseline for any of the three LLMs.
- Vanilla RAG was the most accurate method on HotpotQA for every model (0.60, 0.56, 0.68).
- Conflict-preserving CAS cut the mean prompt from about 1,700–1,770 tokens to roughly 28% of that, at an accuracy cost of 0.03–0.11.

**Conflict surfacing.** On WikiContradict, conflict notes did not increase the rate at which models flagged a conflict above simply showing both passages:
- Vanilla RAG: 0.48–0.80.
- Conflict-preserving CAS: 0.32–0.58.
- Every significant difference was a drop.

**Detection and cost.**
- Pooled over 120 labelled pairs (WikiContradict contradictions and consistent HotpotQA pairs), DeBERTa detected contradictions with F1 0.652. The keyword verifier detected none.
- On RAGTruth, sentence-level entailment did not detect hallucinations better than chance (AUROC 0.55 [0.37, 0.74]).
- The NLI step cost 23 s per query on CPU for 10 passages and 84 s for 20.

### 5.2 Study 1: entity substitution (pre-registered)

**Table 2. Pre-registered decisions** (primary method: conflict-preserving CAS).

| Study | Condition | Comparator | Decision |
|---|---|---|---|
| 1 | A | Vanilla RAG | not supported |
| 1 | A | BM25 top-3 | not supported |
| 1 | B | Vanilla RAG | not supported |
| 1 | B | BM25 top-3 | not supported |
| 1 | B | Provenance + BM25 top-3 | not supported |
| 2 | A | Vanilla RAG | not supported |
| 2 | A | BM25 top-3 | not supported |
| 2 | B | Vanilla RAG | not supported |
| 2 | B | BM25 top-3 | not supported |
| 2 | B | Provenance + BM25 top-3 | not supported |
| 3 | — | H1, evidence F1 | **[S3]** |
| 3 | — | H2, accuracy | **[S3]** |
| 3 | — | H3, non-inferiority | **[S3]** |

**The attack was weak.** Many substitutes were implausible, so poison adoption was near its floor even without filtering (Vanilla RAG: 3.5% for Llama, 6.5% for Gemma). That left little room to show a reduction.

**Accuracy.**
- Vanilla RAG was the most accurate method (Llama 0.515, Gemma 0.395).
- Conflict-preserving CAS was significantly less accurate than Vanilla RAG for Llama in condition A (−0.115 [−0.170, −0.060], Holm p = 0.004).

**Source filter.** In condition B, the filter gave Gemma 1.0% adoption, against 8.0% for conflict-preserving CAS (Holm p = 0.030).

**Preserve vs penalty (exploratory family).** Conflict preservation was more accurate than the penalty, by +0.065 to +0.070 (Holm p ≈ 0.03).

All five decisions were "not supported" (Table 2).

### 5.3 Study 2: a stronger attack (pre-registered)

**The attack was effective.** Without filtering, both models gave the poison answer in 28.5% of questions, and accuracy fell to 0.235 (Llama) and 0.220 (Gemma).

**Condition A (no metadata).** Conflict-preserving CAS did not differ from Vanilla RAG or BM25 top-3 on either endpoint (all Holm p = 1.0). A poison passage reached its context for every question.

**Condition B (noisy metadata).**
- The source filter cut adoption to 3.5% (Llama) and 4.0% (Gemma), with accuracy not significantly different from conflict-preserving CAS.
- Conflict-preserving CAS stayed at 23.0% and 28.0% (differences +0.195 and +0.240, both Holm p = 0.002).
- CAS w/o graph still selected a poison passage for 99.5% of questions.

**Penalty mode** gave the lowest CAS adoption (18.5–21.5%), with no accuracy gain and abstention of up to 20.5–28.5%.

All five decisions were "not supported".

### 5.4 Study 3: clean replication at n = 500 (pre-registered)

**[S3: Report exactly, from `evaluation/results/clean500_summary.json`:**
- **H1:** evidence F1 difference vs Vanilla RAG, BM25 top-3 and Reranker top-3, with 95% CIs and Holm p;
- **H2:** accuracy difference vs the same three comparators, per model;
- **H3:** the CI lower bound vs −0.05, per model;
- **the three decisions;**
- **descriptive:** prompt tokens and abstention.

**No sentence in this section is to be written before the file exists.]**

### 5.5 Study 4: what drives the outcome (pre-registered Monte Carlo)

**Parameters (E1).** Across 1,250 random weight-and-threshold settings, the comparisons with Vanilla RAG and with the source filter had the same sign as at the default setting in 100% of settings (Table 3).
- Conflict-preserving CAS let poison through silently less often than Vanilla RAG (pooled mean difference −0.22 without metadata, −0.27 with noisy metadata).
- It let poison through more often than the source filter (+0.50 [+0.47, +0.52]).
- **The study 2 result is therefore not an artefact of the hand-set weights.**
- Against BM25 top-3 the sign was not stable: CAS was better in 36% and 62% of settings.

**Table 3. Monte Carlo robustness** (conflict-preserving CAS minus comparator; pooled, n = 400 questions; 1,250 random settings).

| Condition | Comparator | Endpoint | Default | Mean over settings [95% CI] | Settings where CAS is better |
|---|---|---|---|---|---|
| A | Vanilla RAG | Silent exposure | −0.213 | −0.220 [−0.254, −0.188] | 100% |
| A | BM25 top-3 | Silent exposure | +0.027 | +0.020 [−0.013, +0.054] | 36% |
| A | BM25 top-3 | Answer passage kept | +0.177 | +0.187 [+0.148, +0.228] | 89.8% |
| B | Vanilla RAG | Silent exposure | −0.255 | −0.267 [−0.301, −0.234] | 100% |
| B | Source filter | Silent exposure | +0.512 | +0.497 [+0.469, +0.524] | 0% |
| B | Source filter | Answer passage kept | +0.188 | +0.174 [+0.148, +0.200] | 91.2% |

**Metadata quality (E2).**
- The source filter reached ≤ 5% silent exposure only with perfectly labelled sources (separation ≥ 0.4). With 10% of sources mislabelled, it bottomed out at 12%.
- Conflict-preserving CAS barely responded to metadata quality: silent exposure stayed between 0.72 and 0.79 across the whole grid. Provenance carries a weight of only 0.25, so it cannot outweigh relevance.

**Detector quality (E3).**
- **Without metadata,** conflict-preserving CAS matched BM25 top-3 once the detector flagged 30% of true conflict pairs. That is about 50% recall per question, roughly five times the current detector's 9.75% (pooled over studies 1–2). The false-alarm rate (0–50%) barely mattered for this endpoint.
- **With noisy metadata,** even a perfect detector left CAS behind the source filter (0.33 vs 0.23).
- **Penalty mode** reached zero silent exposure with a perfect detector, but only by penalising both passages of every flagged pair. The answer passage then survived in 11–17% of questions, against 71–77% with no detection.

**Conflict notes (E4, exploratory).** In the real study 1–2 outputs, a conflict note on the poison did not lower adoption:
- Llama 3.2: 15.7% with a note (n = 70) vs 17.2% without (n = 594).
- Gemma 3: 28.6% vs 22.4%.

### 5.6 Can the detector be made practical? (exploratory)

We evaluated 110 detector configurations, spanning:
- the NLI model: DeBERTa-v3 or MiniLM;
- the unit compared: whole passages or the best sentence;
- θ, a lexical-overlap gate, and one or both directions;
- answer-level detection with two extractive QA models.

**Selection rule.** The rule was fixed in advance: maximise study 1 recall with at most 10% false alarms. Study 2 served as the held-out check.

**Findings.**
- **Current detector:** it caught 14.5% of held-out conflicts and warned falsely on 52.5% of normal questions.
- **Best possible setting:** no configuration exceeded 14% held-out recall at ≤ 10% false alarms.
- **Cost:** the cheapest settings ran in 0.04 s per query, about 500 times faster than the current detector. **Speed is not the obstacle; detection quality is.**

## 6. Discussion

### 6.1 Why CAS does not stop poisoning

Three mechanisms, each measured directly, explain the negative results.

**1. Agreement rewards the poison.**
- A poisoned copy differs from its source in one entity. The NLI model labelled 470 of 845 such pairs (56%) as entailment, and only 67 (8%) as contradiction.
- Poison passages also "agree" with each other (301 of 645 pairs).
- The agreement term (weight 0.20) therefore raises the poison's trust. With no contradiction flags at all, NLI-based CAS let poison through silently in 84% of questions, against 77% without the graph.
- Agreement-based trust assumes that agreement is evidence of truth. PoisonedRAG-style attacks violate this assumption by construction [9].

**2. Provenance is a soft weight.**
- With a weight of 0.25, provenance cannot remove a highly relevant passage, and an attacker controls relevance.
- A hard filter on the same noisy scores outperformed CAS in 100% of weight settings and at every metadata quality.
- When reliability metadata exist, they should gate passages, not be averaged with relevance.

**3. Detection is the binding constraint, and penalties destroy evidence.**
- Today's detector would need about five times its recall for CAS to match BM25 without metadata. No configuration we tested came close at a tolerable false-alarm rate.
- Penalty mode only looks safe with a better detector, because it discards both sides of a conflict, including the answer.

### 6.2 What does work

- **Conflict preservation is the better synthesis mode.** It beat the penalty in accuracy under poisoning (study 1, exploratory), and it did not empty the context on clean data. Once contradictions are flagged, it keeps the answer passage far more often than the penalty: 45–58% vs 11–17% with a perfect detector.
- **On clean retrieval, CAS-style filtering improves evidence precision and cuts prompt length by about 3.5×.** This matters for latency-bound deployments, even though answer accuracy does not improve.

**[S3: state whether the n = 500 replication confirms either point.]**

### 6.3 Design recommendations

1. **Do not treat agreement among retrieved passages as evidence of truth.** At minimum, discount agreement between near-duplicates, and agreement involving passages from the same source.
2. **Use reliability metadata as a gate.** Fall back to trust weighting only among passages that pass it.
3. **Keep both sides of a detected conflict rather than penalising.** But do not expect small (3–4B) generators to act on conflict notes (E4); generators may need training to use them.
4. **Invest in detection recall before detection speed.** Without metadata, CAS needs roughly 50% question-level recall to beat BM25 at the selection level. CPU-cheap detectors already exist, at 0.04 s per query.

## 7. Threats to Validity

**Internal validity.**
- θ was chosen on the 65 clean questions and frozen. The weights and τ are untuned defaults; study 4 shows that the conclusions do not depend on them.
- In study 2, the substitute rules were amended once, before any method ran (documented in the plan).
- Some substitutes remain semantically mismatched (for example, a person's name replacing a place). This makes the attack easier to resist, not harder.

**Construct validity.**
- Accuracy and adoption are string-match metrics. Human validation is in progress (**[HV]**).
- Silent exposure (study 4) is a selection-level proxy. E4 suggests that, for small models, a conflict note does not change adoption, so the proxy may overstate the value of marking.

**External validity.**
- All generators are 3–8B open models run on CPU. Larger models may use conflict notes differently.
- The poisoning studies use HotpotQA only.
- The provenance scores are synthetic, because no benchmark we could access carries real source-reliability labels.
- Self-RAG [10] and CRAG [11] were not re-implemented.

**Statistical conclusion validity.**
- The clean comparison has n = 65 and is exploratory; study 3 replicates it at n = 500.
- The confirmatory families are Holm-corrected.
- The Monte Carlo study reports question-level CIs only. Its run count reduces simulation noise, not sampling error.

**Pre-registration.**
- The plans are hash-stamped in the outputs but were not deposited in a public registry.
- Git history independently confirms that the plan came first only for studies 2 and 3. For studies 1 and 4, the plan first appears in the same commit as the benchmark or the outputs.
- Future studies should be registered publicly before data generation.

## 8. Conclusion

Conflict-aware pre-generation verification is appealing, but in this evaluation it did not deliver its main promise. Across three pre-registered studies of two poisoning attacks, it never reduced poison adoption relative to Vanilla RAG. It was clearly worse than a plain source filter. A Monte Carlo study of more than three million runs showed this is not a matter of tuning. The agreement signal rewards poisoned near-duplicates, soft provenance cannot outweigh relevance, and current NLI detectors miss over 90% of the conflicts that matter.

What survives is narrower, and still useful:
- conflict preservation beats contradiction penalties;
- filtering shortens prompts substantially;
- the requirements for a detector that would make conflict-aware verification work are now quantified.

## Data and Code Availability

All code, pre-registrations, benchmarks, cached NLI and LLM outputs, per-question result files and figure data are available at **[repository URL]**. Every figure is regenerated by `journal_figures/make_all.py`, which checks plotted values against the result files.

## Declarations

**Funding:** [ ]

**Competing interests:** [ ]

**Author contributions:** [ ]

**Use of generative AI:** [Describe the AI assistance used for code, analysis and drafting, as the target journal requires.]

## Figures (manuscript → repository file)

| Manuscript figure | Content | Repository file |
|---|---|---|
| Fig. 1 | CAS pipeline | **[to be drawn]** |
| Fig. 2 | Clean evidence selection; paired differences | `fig01_precision_recall_tradeoff`, `fig02_paired_differences` |
| Fig. 3 | LLM answer quality and cost | `fig08_llm_answer_quality`, `fig10_llm_cost_tradeoff` |
| Fig. 4 | Study 1 | `fig12_poison_study1` |
| Fig. 5 | Study 2 | `fig13_poison_study2` |
| Fig. 6 | Study 3 | **[S3: fig16, after the run]** |
| Fig. 7 | Detector accuracy vs false alarms vs cost | `fig14_detector_tradeoff` |
| Fig. 8 | Monte Carlo study | `fig15_montecarlo` |

## References

*Written from memory. Check every entry against the publisher's record before submission.*

[1] P. Lewis et al., "Retrieval-augmented generation for knowledge-intensive NLP tasks," in *Proc. NeurIPS*, 2020.
[2] Y. Gao et al., "Retrieval-augmented generation for large language models: A survey," arXiv:2312.10997, 2023.
[3] F. Shi et al., "Large language models can be easily distracted by irrelevant context," in *Proc. ICML*, 2023.
[4] N. F. Liu et al., "Lost in the middle: How language models use long contexts," *Trans. ACL*, 2024.
[5] S. Longpre et al., "Entity-based knowledge conflicts in question answering," in *Proc. EMNLP*, 2021.
[6] H.-T. Chen, M. J. Q. Zhang, E. Choi, "Rich knowledge sources bring complex knowledge conflicts: Recalibrating models to reflect conflicting evidence," in *Proc. EMNLP*, 2022.
[7] J. Xie et al., "Adaptive chameleon or stubborn sloth: Revealing the behavior of large language models in knowledge conflicts," in *Proc. ICLR*, 2024.
[8] R. Xu et al., "Knowledge conflicts for LLMs: A survey," in *Proc. EMNLP*, 2024.
[9] W. Zou, R. Geng, B. Wang, J. Jia, "PoisonedRAG: Knowledge corruption attacks to retrieval-augmented generation of large language models," in *Proc. USENIX Security*, 2025.
[10] A. Asai et al., "Self-RAG: Learning to retrieve, generate, and critique through self-reflection," in *Proc. ICLR*, 2024.
[11] S.-Q. Yan et al., "Corrective retrieval augmented generation," arXiv:2401.15884, 2024.
[12] C. Xiang et al., "Certifiably robust RAG against retrieval corruption," arXiv:2405.15556, 2024.
[13] F. Wang et al., "Astute RAG: Overcoming imperfect retrieval augmentation and knowledge conflicts for large language models," arXiv:2410.07176, 2024.
[14] Y. Hou et al., "WikiContradict: A benchmark for evaluating LLMs on real-world knowledge conflicts from Wikipedia," in *Proc. NeurIPS Datasets and Benchmarks*, 2024.
[15] C. Niu et al., "RAGTruth: A hallucination corpus for developing trustworthy retrieval-augmented language models," in *Proc. ACL*, 2024.
[16] P. He, J. Gao, W. Chen, "DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing," in *Proc. ICLR*, 2023.
[17] A. Williams, N. Nangia, S. Bowman, "A broad-coverage challenge corpus for sentence understanding through inference," in *Proc. NAACL*, 2018.
[18] B. A. Nosek et al., "The preregistration revolution," *Proc. Natl. Acad. Sci. USA*, 2018.
[19] S. Robertson, H. Zaragoza, "The probabilistic relevance framework: BM25 and beyond," *Found. Trends Inf. Retr.*, 2009.
[20] Z. Yang et al., "HotpotQA: A dataset for diverse, explainable multi-hop question answering," in *Proc. EMNLP*, 2018.
[21] H. Trivedi et al., "MuSiQue: Multihop questions via single-hop question composition," *Trans. ACL*, 2022.
[22] R. Nogueira, K. Cho, "Passage re-ranking with BERT," arXiv:1901.04085, 2019.
[23] W. Wang et al., "MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers," in *Proc. NeurIPS*, 2020.
[24] A. Grattafiori et al., "The Llama 3 herd of models," arXiv:2407.21783, 2024.
[25] Gemma Team, "Gemma 3 technical report," arXiv:2503.19786, 2025.
[26] P. Rajpurkar et al., "SQuAD: 100,000+ questions for machine comprehension of text," in *Proc. EMNLP*, 2016.
[27] J. Cohen, "A coefficient of agreement for nominal scales," *Educ. Psychol. Meas.*, 1960.
[28] B. Efron, R. J. Tibshirani, *An Introduction to the Bootstrap*. Chapman & Hall, 1993.
[29] S. Holm, "A simple sequentially rejective multiple test procedure," *Scand. J. Statist.*, 1979.
