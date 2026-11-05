# When Does Conflict-Aware Evidence Verification Help Retrieval-Augmented Generation? Five Pre-Registered Studies Under Clean and Poisoned Retrieval

*Draft v0.2, 2026-11-04. Placeholders in **[brackets]** must be filled from result files that do not yet exist. No placeholder may be filled by hand: each names the file and field it comes from.*

---

## Abstract

**Background.** Retrieval-augmented generation (RAG) passes retrieved passages to a language model, including passages that are irrelevant, contradictory or deliberately poisoned. Conflict-aware synthesis (CAS) is a pre-generation verification layer that addresses this. It scores each passage for relevance, source provenance, retrieval rank and agreement with other passages. It uses a natural-language-inference (NLI) model to detect contradictions. It then either penalises contradicting passages or keeps both sides and labels the conflict.

**Methods.** We ran an exploratory comparison and five pre-registered studies, all on CPU only, with parameters that were never tuned on test data. Each pre-registered study had a hash-stamped analysis plan, written before its data were generated.
- **Exploratory, clean retrieval:** HotpotQA, MuSiQue, WikiContradict and RAGTruth, with three local LLMs (3–8B parameters).
- **Studies 1 and 2:** two pre-registered poisoning studies on 400 new HotpotQA questions. One used entity substitution; the other used a stronger, PoisonedRAG-style attack.
- **Study 3:** a pre-registered 500-question replication on clean retrieval, with a cross-encoder reranker baseline.
- **Study 4:** a pre-registered Monte Carlo study of the selection layer (3,064,800 runs), which tested sensitivity to the trust weights, source-metadata quality and detector quality.
- **Study 5:** a pre-registered test of CAS v2, a redesign built from the diagnosis of studies 1–4. It ran on 350 fresh HotpotQA questions under clean retrieval, the study 2 attack and an attack written by a separate LLM.

**Results.**
- On clean retrieval, conflict-preserving CAS selected more precise evidence than Vanilla RAG (F1 +0.299, Holm p = 0.015). It cut prompts to about 28% of Vanilla's length. It did not improve answer accuracy for any model.
- None of the ten pre-registered decisions in poisoning studies 1 and 2 supported CAS. Under the stronger attack, a plain source filter cut poison adoption to 3.5–4.0%; conflict-preserving CAS left it at 22.5–28% (Holm p = 0.002).
- At n = 500, CAS selected significantly better evidence than Vanilla RAG and BM25 (F1 +0.334 and +0.029), but not significantly better than a cross-encoder reranker. It did not improve answer accuracy, and for one model it was significantly less accurate than Vanilla RAG, so all three pre-registered hypotheses were not supported. In exploratory analyses, CAS gave fewer wrong answers than Vanilla RAG (−0.086 and −0.090), mostly by abstaining when its selection lacked the answer.
- The Monte Carlo study shows that these outcomes do not depend on the hand-set weights: the comparisons with Vanilla RAG and with the source filter went the same way in 100% of 1,250 random settings. It also identifies two causes:
  - The NLI model labels 56% of poisoned copies as *agreeing* with their source, so CAS's agreement score rewards the poison.
  - Without source metadata, the detector would need about five times its current recall before CAS matches a BM25 baseline.
- In study 5, CAS v2 met five of nine pre-registered hypotheses.
  - **Clean retrieval:** it kept every supporting passage for 91% of questions (v1: 48.5%) and selected better evidence than all four comparators (F1 +0.096 to +0.406, Holm p = 0.002). It was more accurate than CAS v1 (+0.135 for both models) and non-inferior to Vanilla RAG. Its accuracy gain over Vanilla RAG (+0.065 and +0.055) was not significant after Holm correction.
  - **Copy attack:** without metadata, it stopped the attack entirely (0% adoption, against 27–34% for Vanilla RAG), but only by discarding the true passage along with its copies.
  - **With noisy metadata:** it matched a plain source filter but did not beat it.
  - **Written attack:** it cut adoption by 9 points, against our stated prediction. Post hoc analysis traced the gain to the attacker repeating itself.

**Conclusion.** Conflict-preserving verification is a better design than contradiction penalties. But with current NLI detectors, trust weighting does not protect RAG against poisoning; a hard source filter does. A redesign that follows this diagnosis removed CAS's accuracy loss on clean retrieval, with selection taking 3 ms per question instead of 20 s. It defended only against copy-based poisoning, and then by abstaining. We give the detector and metadata conditions under which conflict-aware verification could pay off.

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

This paper reports such a test. We make five contributions:

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
   - Dropping one supporting passage costs more accuracy than keeping a distractor.
5. **A redesign that follows the diagnosis (CAS v2), tested in a fifth pre-registered study on fresh questions.** The study includes an attack outside the family that v2 was designed against, and a prediction, stated in advance, that v2 would fail against it.

The headline result for the original design is negative. On clean retrieval, CAS improves evidence precision and shortens prompts. It does not improve answers, and it does not protect against poisoning; a simple source filter does. We think this is worth publishing because the failure is specific and measurable, and because it is easy to fix in part.

The redesign shows how far the fixes go. CAS v2 removed the accuracy loss and almost all of the cost, and it stopped copy-based poisoning. But it stopped copies only by abstaining, and it did not beat a source filter when metadata existed. Section 6 states what a detector would have to achieve for conflict-aware verification to pay off.

## 2. Related Work

**Retrieval-augmented generation.** RAG [1] and its many variants [2] condition generation on retrieved text. Later systems add self-critique of retrieved passages (Self-RAG [10]) or a lightweight retrieval evaluator that triggers corrective actions (CRAG [11]). These methods need trained components, and they are not evaluated here; Section 7 discusses this. Long and noisy contexts degrade LLM answers [3], [4], which motivates filtering before generation.

**Knowledge conflicts.** Longpre et al. [5] built entity-substitution conflicts between retrieved context and model memory. Chen et al. [6] studied conflicts among multiple retrieved sources. Xie et al. [7] showed that LLMs accept coherent counter-memory evidence. Xu et al. [8] survey the area. WikiContradict [14] collects real contradictions from Wikipedia, and RAGTruth [15] annotates hallucinations in RAG outputs. We use both.

**Corpus poisoning.** PoisonedRAG [9] showed that a few injected passages, written to be retrieved and to state a target answer, can control RAG output. Certified and aggregation-based defences have been proposed in response [12], [13]. Our study 2 follows the black-box PoisonedRAG design: several mutually consistent poison passages per question, which outnumber the true passage.

**NLI as a verifier.** Cross-encoder NLI models such as DeBERTa-v3 [16], trained on MultiNLI [17] and similar corpora, are widely used to check consistency between texts. Their behaviour on near-duplicate passages that differ in a single entity is central to our results (Section 6.3).

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

### 3.4 CAS v2

CAS v2 (`evaluation/cas_v2.py`) was designed after studies 1–4, using only their data. It changes one thing for each measured cause of CAS's failures (Section 6). In study 5, the original design is called CAS v1: conflict-preserving CAS-NLI with θ = 0.99.

| Step | Rule | Cause addressed |
|---|---|---|
| Gate | When source metadata exist, passages with provenance < 0.5 are removed. | Provenance as a soft weight could not outweigh relevance |
| Quarantine | Two passages are *copies* when ≥ 80% of the shorter one's content words align, in order, with the other (κ = 0.8, at least 5 content words). A copy is *edited* when a run of 1–8 content words is changed, inserted or deleted inside it. Copies are grouped transitively. Every passage in a group that contains an edited copy is removed. | Near-copies "agreed" with their source, and the NLI graph caught under 10% of conflicts |
| Trust | 0.35·*r* + 0.25·*p* + 0.20·*c* ≥ τ = 0.45: v1's weights and τ, without the agreement term | Agreement rewarded the poison |
| Links | A remaining passage whose title (without a parenthetical) is named in the question or in a kept passage is kept too. | Dropping one supporting passage was costly |

v2 has no NLI graph and no conflict notes. The quarantine removes *both* sides of an edit dispute, because the passages alone cannot tell the original from the copy. Section 6.9 shows what that costs.

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
| HotpotQA-5, clean (study 5) | Pre-registered test of CAS v2 | 200 | 10 |
| HotpotQA-5, poisoned (study 5) | Pre-registered test of CAS v2, two attacks | 150 | 13 |

The poisoning and replication samples are drawn from the HotpotQA dev set. They exclude every question used earlier, and the four samples do not overlap.

**Study 1 (entity substitution, after [5]).**
- Each question gets one copy (155 questions) or two copies (45) of a gold passage.
- In each copy, the answer is replaced by another question's answer.
- Questions with yes/no answers or answers longer than five words are excluded.

**Study 2 (PoisonedRAG-style, after [9]).** Each question receives three poison passages. All three state the same type-matched substitute: a shifted year or date, a scaled number, or a title-matched entity. The three forms are:
1. the substituted gold paragraph;
2. the question followed by the substituted paragraph;
3. the title followed by the substituted answer sentences.

The substitute rules were amended once (Amendment 1), before any method ran.

**Study 5 (fresh questions, two attacks).** 200 clean and 150 poisoned questions, drawn with seed 42. Each poisoned question gets one type-matched substitute (the study 2 rule) and is attacked twice:
- **Copy attack:** the study 2 attack.
- **Written attack**, after the black-box PoisonedRAG attack [9]. Three passages are added, each the question followed by a text of at most 30 words. The texts were written by an attacker LLM to make the substitute the answer.
  - **Attacker:** Llama 3.1 8B, temperature 0.8, which is not one of the answering models.
  - **Prompt:** PoisonedRAG's prompt, with two added sentences: the substitute must appear verbatim, and the reply must be the text only.
  - **Acceptance:** a text was used only if it contained the substitute and not the gold answer.
  - **Skipped questions:** 30 of the 180 sampled questions that had a substitute (17%) were skipped, because the attacker did not produce three usable texts within ten samples.

**Source metadata.** Each poisoning study runs in two conditions:
- **Condition A:** no metadata; every passage has provenance 0.5.
- **Condition B:** noisy metadata. Trusted sources score U(0.55, 0.95), untrusted sources U(0.05, 0.45), and each passage's trusted/untrusted status is flipped with probability 0.2.

### 4.2 Methods compared

- **Vanilla RAG:** keeps all passages.
- **BM25 top-3.**
- **Reranker top-3** (studies 3 and 5): `cross-encoder/ms-marco-MiniLM-L-6-v2` [22], [23].
- **Provenance + BM25 top-3** (condition B; "source filter"): a hard source filter, provenance ≥ 0.5, then BM25 top-3.
- **CAS:** w/o graph, keyword, NLI penalty, and NLI conflict-preserving. Conflict-preserving CAS is the primary method in studies 1–4.
- **CAS v2** (Section 3.4): the primary method in study 5.

### 4.3 Generation

- **Models:** Llama 3.2 3B and Gemma 3 4B in all studies, plus Llama 3.1 8B on the clean benchmark [24], [25]. Llama 3.1 8B also wrote the study 5 attack texts.
- **Runtime:** Ollama on CPU, temperature 0, seed 42, context window 8,192 tokens, at most 96 generated tokens.
- **Software:** Python 3.14.6, PyTorch 2.15.0 (CPU build), Sentence-Transformers 6.2.0 and Transformers 5.21.0. These were installed on 2026-10-29, before the first NLI prediction appears in the repository, and were not changed afterwards.
- **Ollama version:** the version was not stored with each cached response; this account comes from Ollama's server logs.
  - 0.40.12 served every answer generated on 2026-10-29 and 2026-10-30: the exploratory comparison and studies 1–2.
  - Ollama updated itself to 0.40.13 on 2026-11-03, before study 3's answers were generated. That version served studies 3 and 5.
  - The logs show no generation requests before 2026-10-29.
  - The model weights did not change between studies. The Ollama model IDs are `a80c4f17acd5` (Llama 3.2 3B), `a2af6cc3eb7f` (Gemma 3 4B) and `46e0c10c039e` (Llama 3.1 8B).
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
  - studies 3 and 4: 2026-11-03;
  - study 5: 2026-11-04.
- **What the git history independently confirms:**
  - Study 2: the plan was committed before its benchmark.
  - Study 3: the plan was committed before any output. The plan shares a commit with the 500-question sample, which a seeded script draws deterministically from the plan's own rules.
  - Study 5:
    - A first version of the plan, with hypotheses H1–H8, was committed before any study 5 data existed.
    - The final version adds H9, the attack-text rules (prompt, extraction, pilot) and the development results. It first appears in a later commit, together with the first 514 attacker outputs. That commit precedes the benchmark file, every selection and every answer.
    - The plan states that the final version was completed before the attack generation started.
  - Studies 1 and 4: the plan first appears in the same commit as the benchmark (study 1) or the outputs (study 4). For these two, the timing rests on the plan's stated date.
- Each output file stores the SHA-256 of its plan. The stored hashes match the files:

| Study | Plan hash (prefix) |
|---|---|
| 1 | 47224db4 |
| 2 | a06a32f1 |
| 3 | c1eb4a89 |
| 4 | 2038df84 |
| 5 | c91b8c27 |

- **Confirmatory families:**
  - studies 1–2: 20 tests each (2 models × 5 comparisons × 2 endpoints), five mechanical decisions;
  - study 3: 9 tests plus a non-inferiority margin of −0.05;
  - study 5: 20 tests plus a non-inferiority margin of −0.05 (Section 4.7).
- **The plans were not deposited in a public registry.** We release the full git history.

**Deviation: scoring corrections.**
- **What happened.** After studies 1–3 had run, a code review found that the answer scoring did not implement the pre-registered definitions. For example, a gold answer "no" was found inside "unknown". The fixes:
  - answers now match as whole words;
  - an abstention is never counted as correct;
  - a fake answer that contains the real answer is scored separately;
  - the exploratory conflict flag needs explicit conflict language.
- **How it was corrected.** All studies were re-scored from the same cached model outputs; no model was re-run. The as-run results are archived.
- **Impact.** No pre-registered decision changed. One confirmatory test became significant, and it went against CAS (study 3, Gemma 3 vs Vanilla RAG: −0.068, Holm p = 0.010). Every number in this paper uses the corrected scoring. The full before/after comparison is in the released `DEVIATIONS.md`.

### 4.6 Monte Carlo design (study 4)

Study 4 re-implements the selection layer in vectorised form. It uses the real BM25 scores and the cached DeBERTa predictions for the 400 questions of studies 1–2, so no model inference is run.

**Validation gate.** Before any simulation, the re-implementation had to reproduce every stored selection and conflict note of studies 1–2. It reproduced all 4,400 exactly.

**Experiments.**
- **E1:** 1,250 random weight vectors, Dirichlet(1, 1, 1, 1), and thresholds τ ~ U(0.30, 0.60), plus the default setting. Every setting ran in both conditions: 1,000,800 runs.
- **E2:** a 7 × 6 grid of metadata quality, varying the trusted–untrusted separation and the flip rate: 1,008,000 runs.
- **E3:** a synthetic detector that flags each true conflict with probability *r* and adds false alarms on 0–50% of normal questions: 1,056,000 runs.
- **E4 (exploratory):** whether conflict notes changed adoption in the real study 1–2 outputs.

Draws are averaged within each question before the bootstrap, and no p-values are computed over draws.

### 4.7 Study 5 design

**How v2's settings were chosen.** Only the data of studies 1–3, which had already been analysed, were used.
- κ, the edit size and the link rule were chosen on a grid over those data (`evaluation/dev_cas_v2.py`).
- τ and the trust weights are v1's untuned defaults.
- Before the plan was written, an answer-level check was run on the first 200 study 3 questions (`evaluation/dev_cas_v2_llm.py`). On it, v2 was more accurate than Vanilla RAG by +0.075 for Llama 3.2 and +0.030 for Gemma 3 (unadjusted).
- The quarantine was designed against copy-and-edit poisoning, the attack family of studies 1–2. The written attack is outside that family.

**Methods per setting.** CAS v1 was left out of the written attack and of condition B, to bound CPU time. Its NLI step costs 20–35 s per question, and its failures in those settings were established by studies 1–2.

| Setting | Methods |
|---|---|
| Clean | Vanilla RAG, BM25 top-3, Reranker top-3, CAS v1, CAS v2 |
| Copy attack, A | Vanilla RAG, CAS v1, CAS v2 |
| Copy attack, B | Source filter, CAS v2 |
| Written attack, A | Vanilla RAG, CAS v2 |
| Written attack, B | Source filter, CAS v2 |

**Hypotheses** (CAS v2 against each comparator; two-sided paired permutation tests; Holm correction over 20 tests):
- **H1:** on clean retrieval, v2 has higher evidence F1 than Vanilla RAG, BM25 top-3, the reranker and CAS v1 (4 tests).
- **H2:** it is more accurate than CAS v1 (2 tests, one per model).
- **H3:** it gives fewer wrong answers than Vanilla RAG (2 tests).
- **H4:** it is non-inferior to Vanilla RAG in accuracy. The 95% CI lower bound of the difference must be above −0.05 (2 CIs).
- **H5:** under the copy attack without metadata, it has lower poison adoption than Vanilla RAG and than CAS v1 (4 tests).
- **H6:** under the copy attack with noisy metadata, it has lower adoption than the source filter (2 tests).
- **H7:** under the written attack without metadata, it has lower adoption than Vanilla RAG (2 tests).
- **H8:** under the written attack with noisy metadata, it has lower adoption than the source filter (2 tests).
- **H9:** on clean retrieval, it is more accurate than Vanilla RAG (2 tests).

**Decision rule.** A hypothesis is supported only if every one of its tests is significant (Holm p < 0.05) in the predicted direction, for both models where tested per model.

**Predictions stated in the plan.**
- **Expected to be supported:** H1, H5 and H6, from the development data.
- **Expected not to be supported:** H7, because the written attack makes no copies for the quarantine to find.
- **Uncertain:** H8.
- **H2 and H4:** expected to be supported.
- **H3 and H9:** expected for Llama 3.2 and uncertain for Gemma 3.
- **Expected cost:** under the copy attack without metadata, v2 would lose accuracy, because the quarantine removes the true passage together with its copies.

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
- Vanilla RAG was the most accurate method on HotpotQA for every model (0.56, 0.52, 0.68).
- Conflict-preserving CAS cut the mean prompt from about 1,700–1,770 tokens to roughly 28% of that, at an accuracy cost of 0.05–0.12.

**Conflict surfacing.** On WikiContradict, conflict notes did not increase the rate at which models flagged a conflict above simply showing both passages:
- Vanilla RAG: 0.22–0.32.
- Conflict-preserving CAS: 0.14–0.22.
- No flagging difference was significant after Holm correction.

**Detection and cost.**
- Pooled over 120 labelled pairs (WikiContradict contradictions and consistent HotpotQA pairs), DeBERTa detected contradictions with F1 0.652. The keyword verifier detected none.
- On RAGTruth, sentence-level entailment did not detect hallucinations better than chance (AUROC 0.55 [0.37, 0.74]).
- The NLI step cost 23 s per query on CPU for 10 passages and 84 s for 20.

### 5.2 Study 1: entity substitution (pre-registered)

**Table 2. Pre-registered decisions** (primary method: conflict-preserving CAS in studies 1–3, CAS v2 in study 5).

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
| 3 | — | H1, evidence F1 | not supported (significant vs Vanilla RAG and BM25, not vs the reranker) |
| 3 | — | H2, accuracy | not supported |
| 3 | — | H3, non-inferiority | not supported |
| 5 | Clean | H1, evidence F1 vs all four comparators | **supported** |
| 5 | Clean | H2, accuracy vs CAS v1 | **supported** |
| 5 | Clean | H3, fewer wrong answers than Vanilla RAG | not supported (significant for Llama 3.2 only) |
| 5 | Clean | H4, non-inferiority to Vanilla RAG | **supported** |
| 5 | Clean | H9, accuracy vs Vanilla RAG | not supported |
| 5 | Copy, A | H5, adoption vs Vanilla RAG and CAS v1 | **supported** |
| 5 | Copy, B | H6, adoption vs source filter | not supported |
| 5 | Written, A | H7, adoption vs Vanilla RAG | **supported** (predicted not to be) |
| 5 | Written, B | H8, adoption vs source filter | not supported |

**The attack was weak.** Many substitutes were implausible, so poison adoption was near its floor even without filtering (Vanilla RAG: 3.5% for Llama, 6.5% for Gemma). That left little room to show a reduction.

**Accuracy.**
- Vanilla RAG was the most accurate method (Llama 0.505, Gemma 0.390).
- Conflict-preserving CAS was significantly less accurate than Vanilla RAG for Llama in condition A (−0.110 [−0.165, −0.055], Holm p = 0.012).

**Source filter.** In condition B, the filter gave Gemma 1.0% adoption, against 8.0% for conflict-preserving CAS (Holm p = 0.030).

**Preserve vs penalty (exploratory family).** Conflict preservation was more accurate than the penalty, by +0.060 to +0.065. The difference was significant in three of four cases (Holm p = 0.031–0.044).

All five decisions were "not supported" (Table 2).

### 5.3 Study 2: a stronger attack (pre-registered)

**The attack was effective.** Without filtering, the models gave the poison answer in 28.0% (Llama) and 28.5% (Gemma) of questions, and accuracy fell to 0.230 and 0.220.

**Condition A (no metadata).** Conflict-preserving CAS did not differ from Vanilla RAG or BM25 top-3 on either endpoint (all Holm p = 1.0). A poison passage reached its context for every question.

**Condition B (noisy metadata).**
- The source filter cut adoption to 3.5% (Llama) and 4.0% (Gemma), with accuracy not significantly different from conflict-preserving CAS.
- Conflict-preserving CAS stayed at 22.5% and 28.0% (differences +0.190 and +0.240, both Holm p = 0.002).
- CAS w/o graph still selected a poison passage for 99.5% of questions.

**Penalty mode** gave the lowest CAS adoption (18.5–21.5%), with no accuracy gain and abstention of up to 20.5–28.5%.

All five decisions were "not supported".

### 5.4 Study 3: clean replication at n = 500 (pre-registered)

Study 3 used 500 new HotpotQA questions, with 10 passages each and no poisoning (Fig. 6). All three pre-registered hypotheses were not supported (Table 2).

**Evidence selection (H1).**
- Conflict-preserving CAS reached evidence F1 0.669, against:
  - Vanilla RAG: 0.335;
  - BM25 top-3: 0.640;
  - the cross-encoder reranker: 0.648.
- It was significantly better than Vanilla RAG (+0.334 [+0.316, +0.353], Holm p < 0.001) and BM25 top-3 (+0.029 [+0.016, +0.043], Holm p = 0.002).
- It was not significantly better than the reranker (+0.021 [−0.001, +0.043], Holm p = 0.40). H1 required all three comparisons, so it is not supported.
- **Exploratory:** CAS without the conflict graph selected slightly better evidence than with it (F1 0.680; difference −0.011, Holm p < 0.001). This repeats the clean-benchmark finding that the graph adds nothing to selection.

**Answer accuracy (H2, H3).**
- Vanilla RAG was the most accurate method (Llama 0.502, Gemma 0.456); conflict-preserving CAS reached 0.464 and 0.388.
- No accuracy comparison favoured CAS.
- For Gemma, CAS was significantly less accurate than Vanilla RAG (−0.068 [−0.108, −0.030], Holm p = 0.010).
- Non-inferiority to Vanilla RAG failed for both models: the CI lower bounds were −0.078 and −0.108, against a margin of −0.05.
- **Abstention:**
  - CAS abstained more often: 15.8% vs 3.4% for Llama, and 24.0% vs 8.2% for Gemma.
  - About half of the questions that Vanilla RAG answered correctly and CAS did not were CAS abstentions: 29 of 61 (Llama) and 37 of 70 (Gemma).

**Exploratory.**
- **Preserve vs penalty:** conflict preservation again beat the contradiction penalty, by +0.058 (Llama) and +0.078 (Gemma) in accuracy and +0.112 in evidence F1, all Holm p < 0.001.
- **Cost:** CAS used about 28% of Vanilla RAG's prompt (406 vs 1,433–1,462 tokens) and generated answers about 3.5 times faster (2.6–3.6 s vs 10.3–12.8 s). Generation is not the whole cost: the NLI step added 23.8 s per question, so end to end CAS was slower than Vanilla RAG (Section 6.5).

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
| A | BM25 top-3 | Answer passage kept | +0.177 | +0.189 [+0.150, +0.231] | 89.9% |
| B | Vanilla RAG | Silent exposure | −0.255 | −0.267 [−0.301, −0.234] | 100% |
| B | Source filter | Silent exposure | +0.512 | +0.497 [+0.469, +0.524] | 0% |
| B | Source filter | Answer passage kept | +0.190 | +0.178 [+0.151, +0.203] | 91.8% |

**Metadata quality (E2).**
- The source filter reached ≤ 5% silent exposure only with perfectly labelled sources (separation ≥ 0.4). With 10% of sources mislabelled, it bottomed out at 12%.
- Conflict-preserving CAS barely responded to metadata quality: silent exposure stayed between 0.72 and 0.79 across the whole grid. Provenance carries a weight of only 0.25, so it cannot outweigh relevance.

**Detector quality (E3).**
- **Without metadata,** conflict-preserving CAS matched BM25 top-3 once the detector flagged 30% of true conflict pairs. That is about 50% recall per question, roughly five times the current detector's 9.75% (pooled over studies 1–2). The false-alarm rate (0–50%) barely mattered for this endpoint.
- **With noisy metadata,** even a perfect detector left CAS behind the source filter (0.33 vs 0.23).
- **Penalty mode** reached zero silent exposure with a perfect detector, but only by penalising both passages of every flagged pair. The answer passage then survived in 10–16% of questions, against 70–77% with no detection.

**Conflict notes (E4, exploratory).** In the real study 1–2 outputs, a conflict note on the poison did not lower adoption:
- Llama 3.2: 15.7% with a note (n = 70) vs 16.8% without (n = 594).
- Gemma 3: 28.6% vs 22.2%.

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

### 5.7 Study 5: CAS v2 on fresh questions (pre-registered)

Five of the nine hypotheses were supported (Table 2; Table 6; Fig. 9). One of them, H7, had been predicted *not* to be supported.

**Table 6. Study 5: confirmatory tests** (CAS v2 minus comparator; 95% bootstrap CIs; p is Holm-corrected over the 20 tests; H4 is judged by its CI against the −0.05 margin). H1 compares selections, which do not depend on the model; its one value per comparator is in the Llama 3.2 column.

| H | Setting | Comparator | Endpoint | Llama 3.2 | Gemma 3 |
|---|---|---|---|---|---|
| H1 | Clean | Vanilla RAG | Evidence F1 | +0.406 [+0.379, +0.431], p = 0.002 | — |
| H1 | Clean | BM25 top-3 | Evidence F1 | +0.119 [+0.090, +0.148], p = 0.002 | — |
| H1 | Clean | Reranker top-3 | Evidence F1 | +0.113 [+0.078, +0.145], p = 0.002 | — |
| H1 | Clean | CAS v1 | Evidence F1 | +0.096 [+0.072, +0.121], p = 0.002 | — |
| H2 | Clean | CAS v1 | Accuracy | +0.135 [+0.080, +0.195], p = 0.002 | +0.135 [+0.085, +0.190], p = 0.002 |
| H3 | Clean | Vanilla RAG | Wrong answers | −0.095 [−0.155, −0.040], p = 0.022 | −0.080 [−0.150, −0.015], p = 0.18 |
| H4 | Clean | Vanilla RAG | Accuracy | +0.065 [+0.015, +0.120]: holds | +0.055 [0.000, +0.115]: holds |
| H9 | Clean | Vanilla RAG | Accuracy | +0.065 [+0.015, +0.120], p = 0.18 | +0.055 [0.000, +0.115], p = 0.41 |
| H5 | Copy, A | Vanilla RAG | Adoption | −0.273 [−0.347, −0.200], p = 0.002 | −0.340 [−0.420, −0.267], p = 0.002 |
| H5 | Copy, A | CAS v1 | Adoption | −0.333 [−0.413, −0.260], p = 0.002 | −0.393 [−0.473, −0.320], p = 0.002 |
| H6 | Copy, B | Source filter | Adoption | −0.013 [−0.053, +0.020], p = 1.0 | −0.013 [−0.053, +0.020], p = 1.0 |
| H7 | Written, A | Vanilla RAG | Adoption | −0.093 [−0.147, −0.040], p = 0.013 | −0.093 [−0.153, −0.033], p = 0.033 |
| H8 | Written, B | Source filter | Adoption | −0.007 [−0.040, +0.027], p = 1.0 | −0.013 [−0.040, +0.013], p = 1.0 |

**Clean retrieval: evidence (H1).**
- CAS v2 reached evidence F1 0.744, against:
  - Vanilla RAG: 0.337;
  - BM25 top-3: 0.625;
  - the reranker: 0.631;
  - CAS v1: 0.647.
- Its precision was about the same as v1's (0.647 vs 0.653), but its recall was far higher (0.955 vs 0.738).
- It kept every supporting passage for 91.0% [86.5, 95.0] of questions, against 48.5% for v1 and 58.0–58.5% for the top-3 filters.
- It selected 3.4 passages per question on average.

**Clean retrieval: answers (H2–H4, H9).**
- **Accuracy:** CAS v2 was the most accurate method for both models.
  - Llama 3.2: 0.550, against 0.485 for Vanilla RAG, 0.450 for BM25 top-3, 0.445 for the reranker and 0.415 for v1.
  - Gemma 3: 0.450, against 0.395, 0.350, 0.355 and 0.315.
- **Against v1 (H2):** the gain was significant (+0.135 for both models).
- **Against Vanilla RAG:** it was non-inferior (H4), but the superiority test (H9) was not significant after Holm correction (Holm p = 0.18 and 0.41). It gave fewer wrong answers than Vanilla RAG, significantly only for Llama 3.2 (H3).
- **Abstention:** it abstained on 4.5% and 11.0% of questions, against 1.5% and 8.5% for Vanilla RAG.
- **Exploratory family:** v2 was more accurate than BM25 top-3 (+0.100 for both models; Holm p = 0.009 and 0.019) and than the reranker (+0.105, Holm p = 0.035; +0.095, Holm p = 0.073).

**Copy attack (H5, H6).**
- **Condition A, no metadata.** No CAS v2 answer, for either model, gave the fake answer (0 of 150). A poison passage reached v2's prompt for one question. Vanilla RAG gave the fake answer in 27.3% and 34.0% of questions, and CAS v1 in 33.3% and 39.3%.
- **The cost was the one predicted.**
  - Accuracy fell to 0.073 for both models, against 0.247 and 0.213 for Vanilla RAG (exploratory: −0.173 and −0.140, Holm p = 0.004).
  - Abstention rose to 38.7% and 43.3%.
  - Wrong answers fell, from 0.707 and 0.773 to 0.540 and 0.493 (Holm p = 0.017 and 0.004).
- **Condition B, noisy metadata.** v2's adoption (3.3% for both models) did not differ from the source filter's (4.7%). Accuracy did not differ either: 0.280 vs 0.313 for Llama 3.2, and 0.213 vs 0.247 for Gemma 3; both Holm p = 1.0.

**Written attack (H7, H8).**
- **Condition A, no metadata.** Without filtering, the written attack was about as effective as the copy attack: Vanilla RAG gave the fake answer in 30.7% and 33.3% of questions. It lowered Vanilla RAG's accuracy further, to 0.107 and 0.087. CAS v2 lowered adoption to 21.3% and 24.0% (−0.093 for both models). H7 is therefore supported, against our prediction.
  - A poison passage still reached v2's prompt for 90% of questions.
  - Accuracy did not change significantly (0.147 vs 0.107 and 0.127 vs 0.087).
  - Section 6.9 explains where the reduction came from.
- **Condition B, noisy metadata.** v2 again matched the source filter: adoption 10.7% vs 11.3% and 12.7% vs 14.0%; accuracy 0.240 vs 0.253 and 0.273 vs 0.220.

**Cost.**
- **Selection time:** v2 took 3 ms per question, against 19.8 s for CAS v1 with 10 passages and 34.7 s with 13.
- **Prompt length:** v2's prompts were 34–35% of Vanilla RAG's length (487–488 vs 1,398–1,430 tokens).
- **Generation time:** 2.3 s (Llama 3.2) and 4.1 s (Gemma 3), against 10.1 s and 12.2 s for Vanilla RAG.
- **End to end:** because selection costs almost nothing, v2 was 2.9–4.3 times faster than Vanilla RAG. CAS v1 was slower than Vanilla RAG, because of its NLI step.

## 6. Discussion

The results raise four questions:
- Why did better evidence not give better answers (Sections 6.1–6.2)?
- Why does CAS not stop poisoning (Section 6.3)?
- Why do conflict notes not change answers (Section 6.4)?
- What does the conflict graph cost (Section 6.5)?
- Did fixing the diagnosed causes work (Section 6.9)?

Sections 6.1, 6.2, 6.4 and 6.5 use exploratory analyses of the cached study 3 outputs, chosen after the results were known (`evaluation/run_discussion_analyses.py`, output `results/discussion_analyses.json`). They were not pre-registered. Section 6.9 likewise uses a post hoc analysis of study 5's selections (`evaluation/run_study5_mechanism.py`, output `results/study5_mechanism.json`). These analyses help explain the confirmatory results but do not replace them, and each would need its own confirmatory test.

### 6.1 Why better evidence did not give better answers

In study 3, CAS doubled evidence F1 relative to Vanilla RAG, yet answered less accurately. This is not specific to CAS. Every filter, including BM25 top-3 and the reranker, was less accurate than Vanilla RAG for both models (Section 5.4).

**Filtering helps when it keeps the evidence, and hurts when it drops it.** We split the 500 questions by whether a filter kept every supporting passage. Within each group, we compared the filter with Vanilla RAG on the same questions (Table 4).
- **When a filter kept every supporting passage,** it was *more* accurate than Vanilla RAG in all ten filter–model combinations (+0.039 to +0.062). The 95% CI excluded zero in eight of them. After Holm correction over the 20 stratified tests, only one remained significant: BM25 top-3 with Llama 3.2 (Holm p = 0.044).
- **When a filter dropped a supporting passage,** it was much less accurate (−0.129 to −0.258; all Holm p ≤ 0.003).
- **Most of CAS's losses were in the second group.** Take the questions that Vanilla RAG answered correctly and CAS did not. CAS had dropped a supporting passage on 48 of 61 of them (Llama 3.2) and on 58 of 70 (Gemma 3).

**Table 4. Accuracy by whether the filter kept every supporting passage** (study 3; exploratory; each filter is paired with Vanilla RAG on the same questions).

| Model | Filter | Supporting passages | n | Filter accuracy | Vanilla RAG accuracy | Difference [95% CI] |
|---|---|---|---|---|---|---|
| Llama 3.2 | BM25 top-3 | all kept | 308 | 0.610 | 0.549 | +0.062 [+0.019, +0.101] |
| | | one or more dropped | 192 | 0.250 | 0.427 | −0.177 [−0.250, −0.109] |
| | Reranker top-3 | all kept | 314 | 0.602 | 0.551 | +0.051 [+0.010, +0.092] |
| | | one or more dropped | 186 | 0.194 | 0.419 | −0.226 [−0.296, −0.161] |
| | CAS-NLI preserve | all kept | 259 | 0.622 | 0.575 | +0.046 [0.000, +0.093] |
| | | one or more dropped | 241 | 0.295 | 0.423 | −0.129 [−0.195, −0.062] |
| Gemma 3 | BM25 top-3 | all kept | 308 | 0.542 | 0.487 | +0.055 [+0.006, +0.101] |
| | | one or more dropped | 192 | 0.198 | 0.406 | −0.208 [−0.276, −0.141] |
| | Reranker top-3 | all kept | 314 | 0.519 | 0.468 | +0.051 [+0.003, +0.099] |
| | | one or more dropped | 186 | 0.177 | 0.435 | −0.258 [−0.339, −0.177] |
| | CAS-NLI preserve | all kept | 259 | 0.544 | 0.494 | +0.050 [+0.004, +0.097] |
| | | one or more dropped | 241 | 0.220 | 0.415 | −0.195 [−0.261, −0.137] |

**A distractor costs less than a missing passage.** The two effects agree with what is known about long contexts.
- Irrelevant passages distract LLMs [3], [4]. Removing them gained about five points here.
- But a HotpotQA question needs two passages. A filter that removes one leaves the model unable to answer, however precise the rest of the selection is.
- At ten passages (about 1,450 prompt tokens), distraction cost less than losing a needed passage.

**Evidence F1 rewarded the wrong trade-off.**
- CAS was more precise than BM25 top-3 and the reranker (0.670 vs 0.533 and 0.540), but less complete (recall 0.754 vs 0.799 and 0.809).
- It kept every supporting passage for 51.8% of questions, against 61.6% and 62.8%.
- Its F1 advantage was therefore a precision advantage. The answer depends mainly on whether every needed passage survives.

**What a filter would need.** Suppose the effects in the two groups stayed as measured. Conflict-preserving CAS would then match Vanilla RAG's accuracy only if it kept every supporting passage for about 74% (Llama 3.2) to 80% (Gemma 3) of questions, against 52% today. This extrapolation is rough, because the two groups are not comparable. The questions on which a filter keeps all the evidence are easier: Vanilla RAG, too, is more accurate on them (0.575 vs 0.423 for Llama 3.2).

### 6.2 CAS fails safe: fewer wrong answers, more abstentions

Accuracy counts an abstention and a wrong answer the same way. Table 5 separates them.

**Table 5. Answer outcomes on the 500 clean questions of study 3** (share of questions; exploratory).

| Method | Llama 3.2 correct | Llama 3.2 wrong | Llama 3.2 abstained | Gemma 3 correct | Gemma 3 wrong | Gemma 3 abstained |
|---|---|---|---|---|---|---|
| Vanilla RAG | 0.502 | 0.464 | 0.034 | 0.456 | 0.462 | 0.082 |
| BM25 top-3 | 0.472 | 0.416 | 0.112 | 0.410 | 0.388 | 0.202 |
| Reranker top-3 | 0.450 | 0.450 | 0.100 | 0.392 | 0.390 | 0.218 |
| CAS w/o graph | 0.466 | 0.376 | 0.158 | 0.390 | 0.366 | 0.244 |
| CAS-NLI penalty | 0.406 | 0.342 | 0.252 | 0.310 | 0.364 | 0.326 |
| CAS-NLI preserve | 0.464 | 0.378 | 0.158 | 0.388 | 0.372 | 0.240 |

**Fewer wrong answers.**
- Conflict-preserving CAS gave fewer wrong answers than Vanilla RAG: −0.086 [−0.126, −0.046] for Llama 3.2 and −0.090 [−0.134, −0.048] for Gemma 3. Both have Holm p = 0.001, over ten tests.
- Every filter reduced wrong answers. The reduction was significant in nine of the ten filter–model combinations, and largest for the CAS variants (0.086–0.122).

**The abstentions were mostly justified.** We counted only answers that are text spans, not yes/no answers.
- For CAS, the answer appeared nowhere in its selected passages in 51 of its 72 abstentions (71%, Llama 3.2) and 57 of 100 (57%, Gemma 3). The model was right to say that it could not answer from what it was given.
- Vanilla RAG's abstentions never had this justification. In all 15 and all 32 cases, the answer was in its context.

**Answered questions were not less accurate.**
- Among the questions it did answer, CAS was slightly, not significantly, more accurate than Vanilla RAG:
  - Llama 3.2: 0.551 vs 0.520 (+0.031 [−0.010, +0.071]);
  - Gemma 3: 0.511 vs 0.497 (+0.014 [−0.032, +0.056]).
- It answered fewer questions: 84.2% vs 96.6% (Llama 3.2) and 76.0% vs 91.8% (Gemma 3).

**Whether this is a good trade depends on the cost of a wrong answer.**
- Per 500 questions, compared with Vanilla RAG, CAS gave:
  - Llama 3.2: 19 fewer correct answers, 43 fewer wrong answers and 62 more abstentions;
  - Gemma 3: 34 fewer correct, 45 fewer wrong and 79 more abstentions.
- Score a correct answer +1, an abstention 0 and a wrong answer −*c*. CAS then beats Vanilla RAG whenever *c* > 0.44 (Llama 3.2) or *c* > 0.76 (Gemma 3).
- At *c* = 1, the advantage was +0.048 [−0.026, +0.118] and +0.022 [−0.052, +0.092]. Neither is significant (Holm p = 1.0).
- The "fails safe" reading is therefore suggestive, not established. It deserves a pre-registered test with risk–coverage endpoints, in applications where a wrong answer costs more than no answer, such as clinical, legal or enterprise question answering.

**The safety comes from filtering, not from conflict detection.** CAS without the graph shows the same pattern: wrong answers fell by 0.088 and 0.096. So do BM25 top-3 and the reranker, by less.

### 6.3 Why CAS does not stop poisoning

Three mechanisms, each measured directly, explain the negative results.

**1. Agreement rewards the poison.**
- A poisoned copy differs from its source in one entity. The NLI model labelled 470 of 845 such pairs (56%) as entailment, and only 67 (8%) as contradiction.
- Poison passages also "agree" with each other (301 of 645 pairs).
- The agreement term (weight 0.20) therefore raises the poison's trust. With no contradiction flags at all, NLI-based CAS let poison through silently in 84% of questions, against 77% without the graph.
- This is consistent with a known weakness of NLI classifiers trained on MultiNLI: they lean on lexical overlap, and predict entailment for pairs that share most of their words [30]. A poisoned copy shares all but one entity with its source.
- Agreement-based trust assumes that agreement is evidence of truth. That holds only when the agreeing sources are independent. Near-duplicates and coordinated poison are not independent, and PoisonedRAG-style attacks violate the assumption by construction [9].
- In study 2, three mutually consistent poison passages faced one true passage, so any defence that counts agreement sides with the poison. Certified aggregation defences [12] likewise give their guarantees only for a bounded number of corrupted passages.

**2. Provenance is a soft weight.**
- With a weight of 0.25, provenance cannot remove a highly relevant passage, and an attacker controls relevance.
- A hard filter on the same noisy scores outperformed CAS in 100% of weight settings and at every metadata quality.
- Better metadata did not help CAS. Across the whole metadata-quality grid (E2), its silent exposure stayed between 0.72 and 0.79, because the weight never let the metadata decide.
- When reliability metadata exist, they should gate passages, not be averaged with relevance.

**3. Detection is the binding constraint, and penalties destroy evidence.**
- Today's detector would need about five times its recall for CAS to match BM25 without metadata. No configuration we tested came close at a tolerable false-alarm rate.
- Penalty mode only looks safe with a better detector, because it discards both sides of a conflict, including the answer.

**Why the selection-level gain did not reach the answers.**
- In the Monte Carlo study, CAS let poison through silently less often than Vanilla RAG in all 1,250 settings (mean differences 0.22 and 0.27).
- In the real outputs, it did not significantly reduce adoption.
- The gain consisted of conflict notes and some removals. The notes did not change answers (Section 6.4), and in study 2, condition A, a poison passage still reached CAS's context for every question.

### 6.4 Why conflict notes do not change answers

Three observations point the same way:
1. **WikiContradict.** Notes did not raise the rate at which models flagged a conflict (Section 5.1).
2. **Poisoning.** A note on the poison passage did not lower adoption (E4): 15.7% vs 16.8% for Llama 3.2, and 28.6% vs 22.2% for Gemma 3.
3. **Clean data.**
   - The detector attached notes to 49 of the 500 study 3 questions (100 notes in all), although no conflicts had been planted.
   - On those questions, CAS was not more accurate than CAS without the graph: −0.061 [−0.143, +0.020] for Llama 3.2 and −0.041 [−0.122, +0.041] for Gemma 3.
   - This comparison also includes any difference in selection, not only the note.

**Interpretation.** A 3–4B instruction-tuned model appears to treat "passages [i] and [j] contradict each other" as one more piece of context, not as an instruction to weigh the evidence. This matches the finding that LLMs follow coherent and plentiful evidence, even when other evidence contradicts it [7].

**Caveats.**
- **E4 is observational.** A note was attached only where the detector fired, which may be where the conflict was most obvious. A randomised ablation is needed: one that adds or removes the note with everything else fixed.
- **We tested one wording.** A note that says which source is more reliable, or one that asks the model to report both values, might behave differently.
- **Training is another route.** Generators trained to critique their context, as in Self-RAG [10], may use such signals better.

### 6.5 What the conflict graph costs and buys

- **Cost.** In study 3, building the graph for ten passages took 23.8 s per question on CPU (45 pairs, each scored in both directions).
  - End to end, conflict-preserving CAS took 26.4 s (Llama 3.2) and 27.5 s (Gemma 3) per question, against 10.3 s and 12.8 s for Vanilla RAG. That is 2.2–2.6 times slower.
  - The trust-score filter alone (CAS w/o graph) took 2.5 s and 3.5 s, 3.7–4.1 times faster than Vanilla RAG.
- **Benefit on clean data: none measurable.** Without the graph, evidence F1 was slightly higher (0.680 vs 0.669) and accuracy was the same (difference −0.002 for both models, Holm p = 1.0).
- **Benefit under poisoning.** The graph flagged a true conflict in 9.75% of poisoned questions.
- **Conclusion.** With today's detectors and CPU hardware, the graph is almost pure cost. The prompt and latency savings reported for CAS come from the trust-score filter. A cheap detector (0.04 s per query, Section 5.6) would remove the cost, but none of the configurations we tested would supply the missing recall.

### 6.6 What does work

- **Conflict preservation is the better synthesis mode.**
  - It beat the penalty in accuracy under poisoning (study 1, exploratory), and it did not empty the context on clean data.
  - Once contradictions are flagged, it keeps the answer passage far more often than the penalty: 45–58% vs 10–16% with a perfect detector.
  - Study 3 confirmed the advantage at n = 500: +0.058 to +0.078 accuracy and +0.112 evidence F1 (exploratory, Holm p < 0.001).
- **Trust-score filtering has real benefits.** It improves evidence precision, cuts prompts to about 28% of their length, and gives fewer wrong answers (Section 6.2). Without the conflict graph, it also cuts end-to-end latency on CPU by a factor of 3.7–4.1.
- **A hard source filter works against poisoning.** When reliability metadata exist, it was the one defence that worked: adoption was 3.5–4.0% in study 2.
- **A reranker matches CAS.** At n = 500, a cross-encoder reranker matched CAS's evidence selection without any conflict graph.

### 6.7 Implications for evaluating pre-generation verification

1. **Selection-level metrics overstated the benefit, twice.**
   - Evidence F1 doubled while accuracy fell (Section 6.1).
   - Silent exposure fell in every Monte Carlo setting, while adoption did not significantly fall (Section 6.3).
   - Pre-generation methods should be judged on answers. At the selection level, the share of questions with every supporting passage kept tells more than F1.
2. **Report correct, wrong and abstained answers separately.** Accuracy alone hides the trade shown in Table 5.
3. **Report end-to-end latency, including verification.** Generation time alone made CAS-NLI look about 3.5 times faster than Vanilla RAG. End to end, it was 2.2–2.6 times slower.
4. **Fix the analysis before the data exist.**
   - Pre-registration made the negative results interpretable, because no endpoint could be chosen after the fact.
   - It also let us show that the scoring errors found later changed no decision (Section 4.5).
   - The Monte Carlo study ruled out the most common objection to a negative result: that the parameters were wrong.

### 6.8 Design recommendations

These recommendations were written from studies 1–4. CAS v2 implements recommendations 1 (in its strongest form: no agreement term at all), 2, 5 and 7. Section 6.9 reports how it fared.

1. **Do not treat agreement among retrieved passages as evidence of truth.** At minimum, discount agreement between near-duplicates, and agreement involving passages from the same source.
2. **Use reliability metadata as a gate.** Fall back to trust weighting only among passages that pass it.
3. **Keep both sides of a detected conflict rather than penalising.** But do not expect small (3–4B) generators to act on conflict notes (E4); generators may need training to use them.
4. **Invest in detection recall before detection speed.** Without metadata, CAS needs roughly 50% question-level recall to beat BM25 at the selection level. CPU-cheap detectors already exist, at 0.04 s per query.
5. **Optimise filters for evidence recall, not F1.** For multi-hop questions, a filter should keep every passage the answer needs, even at some cost in precision. One option is to keep any passage that links to a passage already kept.
6. **Pair lossy filtering with abstention, and measure the trade-off.** If a filter can drop needed evidence, the generator should be allowed to abstain. The system should then be judged on risk and coverage, not on accuracy alone.
7. **Leave the conflict graph out until the detector is good enough.** Until a detector reaches roughly 50% question-level recall at a tolerable false-alarm rate, CAS without the graph selects evidence as well, at a fraction of the cost.

### 6.9 Testing the diagnosis: what CAS v2 fixed and what it did not

Study 5 tested four of the recommendations together, on questions that no earlier study had used. The mechanism figures below are post hoc.

**1. Evidence recall was the lever for accuracy.**
- v2 kept every supporting passage for 91% of questions.
- Section 6.1 estimated that a filter would match Vanilla RAG's accuracy at about 74–80%. v2 exceeded that, and it was more accurate than Vanilla RAG for both models (+0.065 and +0.055).
- This agrees with the estimate, but two cautions apply: the gain was not significant after Holm correction (H9), and the estimate came from a different sample.
- **The title links did most of the work.** They added a passage for 61% of clean questions, and 89 of the 159 passages they added were supporting passages.
- **Precision did not suffer:** v2's precision was about the same as v1's (0.647 vs 0.653).
- **Result:** the change from v1 to v2 gained 0.135 in accuracy for both models.

**2. The quarantine stops copies, but cannot tell the original from the copy.**
- Under the copy attack without metadata, the quarantine removed almost every poison passage (2.99 of 3 per question). It also removed the true passage in all 150 questions, because each poison passage is an edited copy of it.
- **This is the penalty-mode failure of Section 6.3 in another form.** Removing both sides of a dispute buys safety at the price of the answer. v2 turned wrong answers into abstentions, not into correct answers:
  - wrong answers −0.17 and −0.28;
  - abstention +0.34 and +0.42;
  - accuracy −0.17 and −0.14.
- **False quarantines:** on clean data, the quarantine fired for 14 of 200 questions, where no poison existed.
- **What would be needed:** separating the original from the copy needs information the passages do not contain, such as provenance, a timestamp or the retrieval source.

**3. When metadata exist, v2 is a source filter.**
- With noisy metadata, v2 matched the source filter under both attacks, in adoption and in accuracy (H6 and H8 not supported).
- **The gate did the work.** It removed 2.47 (copy attack) and 2.43 (written attack) of the three poison passages per question, before the quarantine or the trust score saw them.
- **The quarantine cost a little here.**
  - When a mislabelled poison copy passed the gate, the quarantine removed the true passage with it. This happened for 34% of copy-attack questions.
  - Over all copy-attack questions, the answer was in v2's prompt for 48.7%, against 60.0% for the source filter. The accuracy difference was not significant.
- **A better rule, untested:** let provenance settle a dispute when metadata exist, and quarantine only when they do not.

**4. The written attack: a supported hypothesis, a wrong prediction and a narrow mechanism.**
- We predicted that H7 would fail, because the written attack makes no copies. It was supported: −0.093 for both models.
- **The attacker repeated itself.** It was sampled with one prompt until three texts were accepted, and it often produced near-copies. For 33 of 150 questions, two or three of its texts were near-copies with small edits, and the quarantine removed them (all three for 15 questions, two for 18).
  - **All three texts removed (15 questions):** neither model gave the fake answer. Vanilla RAG gave it on 53% (Llama 3.2) and 60% (Gemma 3) of the same questions.
  - **All three texts reached the prompt (117 questions):** adoption was close to Vanilla RAG's: 22.2% vs 25.6% for Llama 3.2, and 26.5% vs 28.2% for Gemma 3. This held although the poison made up 56% of v2's prompt, against 23% of Vanilla RAG's.
- **So v2 has no defence against diverse written poison.** Its H7 gain depends on the attacker's lack of diversity, which an attacker can remove by deduplicating or paraphrasing its texts.
- **The written attack also displaced the true evidence.**
  - v2 dropped the true passage for 70 of 150 questions.
  - In every one of the 70, the three poison passages, each led by the question, held the top three BM25 ranks. In 69 of the 70, the true passage fell below τ.
  - On the same questions without the poison, v2 kept the true passage for 138 of 150.
  - A question-led attack therefore beats a relevance threshold twice: the poison passes it, and the true passage falls below it.
- **A lesson for evaluation.** A pre-registered decision says whether an effect exists, not why. Without the mechanism analysis, H7 would read as "v2 resists LLM-written poisoning", and that reading is wrong.

**5. Cost.** Without the NLI graph, selection took 3 ms per question instead of 20–35 s, and v2 ran 2.9–4.3 times faster than Vanilla RAG end to end.

**What this means.**
- **On clean retrieval, the diagnosis was right.** A filter that keeps the evidence helps, and dropping the graph removes the cost.
- **On poisoning, the diagnosis was incomplete.**
  - Removing the agreement bonus and quarantining copies handles the attack the diagnosis was built on. But the defence fails safe, and it does not recover the answer.
  - It has no answer to diverse poison that is not a copy.
  - Against such poison, source metadata remained the only effective tool in our studies. Under the written attack, the source filter left 11.3% and 14.0% adoption, against 30.7% and 33.3% for Vanilla RAG. This comparison is descriptive, not a pre-registered test. It is valid because Vanilla RAG ignores metadata and both conditions use the same passages, so Vanilla RAG's condition A results also stand for condition B.

## 7. Threats to Validity

**Internal validity.**
- θ was chosen on the 65 clean questions and frozen. The weights and τ are untuned defaults; study 4 shows that the conclusions do not depend on them.
- In study 2, the substitute rules were amended once, before any method ran (documented in the plan).
- Some substitutes remain semantically mismatched (for example, a person's name replacing a place). This makes the attack easier to resist, not harder.
- **CAS v2 was designed after studies 1–4, on their data, including the copy attack.**
  - H5 therefore tests whether v2 generalises to new questions, not to a new attack.
  - The written attack is the only out-of-family test, and the one test where v2's gain turned out to be incidental.
- **The study 5 mechanism analysis (Section 6.9) is post hoc.**
- **The Ollama runtime changed between studies.** It moved from 0.40.12 to 0.40.13 between study 2 and study 3 (Section 4.3), with the same model weights. Comparisons within a study are unaffected, because every method in a study ran on the same version. Comparisons of absolute accuracy across studies could be affected.

**Construct validity.**
- Accuracy and adoption are string-match metrics. Human validation is in progress (**[HV]**). The same applies to the "wrong" category in Table 5: a correct paraphrase that does not contain the gold string counts as wrong.
- Silent exposure (study 4) is a selection-level proxy. E4 suggests that, for small models, a conflict note does not change adoption, so the proxy may overstate the value of marking.
- **The written attack (study 5) uses one attacker model and one prompt.**
  - The prompt adds two sentences to PoisonedRAG's.
  - 17% of the sampled questions were skipped because the attacker failed, so the attacked set is conditional on the attacker's success.
  - An attacker that diversifies its texts would remove v2's incidental gain (Section 6.9).
  - The attacker, Llama 3.1 8B, belongs to the same model family as one answering model, Llama 3.2 3B.

**External validity.**
- All generators are 3–8B open models run on CPU. Larger models may use conflict notes differently.
- The poisoning studies use HotpotQA only.
- The provenance scores are synthetic, because no benchmark we could access carries real source-reliability labels.
- Self-RAG [10] and CRAG [11] were not re-implemented.
- v2's link rule uses HotpotQA's passage titles. A production retriever may not supply titles in the same form.
- In study 5, CAS v1 was not run under the written attack or with metadata, to bound CPU time.

**Statistical conclusion validity.**
- The clean comparison has n = 65 and is exploratory; study 3 replicates it at n = 500.
- The confirmatory families are Holm-corrected.
- The Monte Carlo study reports question-level CIs only. Its run count reduces simulation noise, not sampling error.
- Study 5's clean set is small (n = 200). Its accuracy differences have 95% CIs about ±0.05 wide, so differences of 0.055–0.065 did not survive Holm correction over 20 tests. The plan anticipated this for Gemma 3 (H3, H9).
- The analyses in Sections 6.1, 6.2, 6.4, 6.5 and 6.9 were chosen after the results were known. Their tests are Holm-corrected within each analysis, but the set of analyses was not fixed in advance. The groups in Table 4 are defined by each method's own selection, so they are not randomly assigned.

**Pre-registration.**
- The plans are hash-stamped in the outputs but were not deposited in a public registry.
- Git history independently confirms that the plan came first only for studies 2 and 3, and partly for study 5.
  - For studies 1 and 4, the plan first appears in the same commit as the benchmark or the outputs.
  - For study 5, git confirms a first version of the plan before any data. It confirms the final version before the benchmark and every answer, but not before the first attacker outputs (Section 4.5).
- Future studies should be registered publicly before data generation.

## 8. Conclusion

Conflict-aware pre-generation verification makes two claims: it selects better evidence, and it protects generation from conflicting or poisoned passages. Our evaluation partly supports the first claim. It also shows precisely why the second fails with current components.

**Clean retrieval.** Conflict-preserving CAS selected significantly more precise evidence than Vanilla RAG and cut prompts to about 28% of their length. Conflict preservation also outperformed contradiction penalties.

**At scale.** In a pre-registered replication on 500 clean questions, CAS selected significantly better evidence than Vanilla RAG and BM25, but not better than a cross-encoder reranker. It did not improve answer accuracy.

**Why, and what it does instead.** Exploratory analyses explain the gap. Filtering helped when it kept every supporting passage and hurt when it dropped one, and CAS dropped one for nearly half of the questions. CAS also gave significantly fewer wrong answers than Vanilla RAG (−0.086 and −0.090), mostly by abstaining when its selection did not contain the answer. Where a wrong answer costs more than no answer, that trade may favour CAS. This needs a confirmatory test.

**Poisoning.** In pre-registered poisoning studies 1 and 2, CAS did not significantly reduce poison adoption relative to Vanilla RAG (differences −0.055 to +0.030, all Holm p = 1.0). When source metadata were available, a plain source filter performed significantly better.

**Causes.** A pre-registered Monte Carlo study of more than three million selection runs traced the poisoning result to three measurable causes. The exploratory analyses added a fourth. Each points to a concrete fix, and study 5 tested the fixes on fresh questions:

| Cause | Fix in CAS v2 | Study 5 result (pre-registered) |
|---|---|---|
| The agreement signal rewards poisoned near-duplicates | No agreement term; edited copies are quarantined | Copy attack stopped (0% adoption, against 27–34% for Vanilla RAG), but the true passage was removed too; accuracy 0.073 |
| Provenance, entered as a soft weight, cannot outweigh relevance | Provenance as a gate | Equal to a plain source filter; no gain over it |
| The NLI detector catches fewer than 10% of the conflicts that matter, at 20–35 s per question | No NLI graph | Selection in 3 ms; v2 faster than Vanilla RAG end to end |
| Dropping a supporting passage costs more than keeping a distractor | Keep passages linked by title | Every supporting passage kept for 91% of questions (v1: 48.5%); accuracy +0.135 over v1, non-inferior to Vanilla RAG, superiority not established |

**What remains open.** The redesign fixed what the diagnosis covered: evidence recall, cost and copy-based poisoning. It did not supply what the diagnosis said was missing.
- **A way to tell the original from the copy.** Without one, v2 defends by abstaining rather than by answering correctly.
- **A defence against diverse, non-copy poison.** v2's gain against the LLM-written attack came only from the attacker repeating itself. Where metadata exist, a source filter does as well as v2.

A conflict-aware verifier that does better than a source filter will need a detector, or provenance signals, that current components do not provide.

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
| Fig. 6 | Study 3 | `fig16_clean500` |
| Fig. 7 | Detector accuracy vs false alarms vs cost | `fig14_detector_tradeoff` |
| Fig. 8 | Monte Carlo study | `fig15_montecarlo` |
| Fig. 9 | Study 5: CAS v2 on fresh questions | `fig17_study5` |

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
[30] R. T. McCoy, E. Pavlick, T. Linzen, "Right for the wrong reasons: Diagnosing syntactic heuristics in natural language inference," in *Proc. ACL*, 2019.
