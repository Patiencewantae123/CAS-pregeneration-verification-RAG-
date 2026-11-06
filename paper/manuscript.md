# From Diagnosis to Confirmation: Lightweight Conflict-Aware Evidence Selection for Retrieval-Augmented Generation in Six Pre-Registered Studies

*Draft v0.3, 2026-11-06. Placeholders in **[brackets]** must be filled from result files that do not yet exist. No placeholder may be filled by hand: each names the file and field it comes from.*

---

## Highlights

- Six pre-registered studies test conflict-aware evidence selection for RAG.
- NLI-based conflict verification failed; four measurable causes were found.
- CAS v2 replaces the NLI graph with a gate, a copy quarantine and title links.
- On 700 fresh questions, CAS v2 beat Vanilla RAG and a reranker in accuracy.
- On MuSiQue, CAS v2 cut wrong answers but did not raise accuracy.

## Abstract

Retrieval-augmented generation (RAG) passes irrelevant, contradictory and poisoned passages to the language model. Conflict-aware verification filters them before generation, but has rarely been tested in a way that allows a negative answer. We evaluated a conflict-aware synthesis framework (CAS) in six pre-registered studies with hash-stamped analysis plans, using 3–4B open models on CPU. Studies 1–4 (900 questions and 3.06 million simulated selections) found that the original design, which detects conflicts with a natural-language-inference (NLI) model, neither improved accuracy nor resisted poisoning. They traced this to four measurable causes: poisoned near-duplicates counted as agreement; provenance was too weak a weight; the NLI detector caught under 10% of conflicts at 20–35 s per question; and dropping a supporting passage cost more than keeping a distractor. We built CAS v2 from this diagnosis. It combines a provenance gate, an edited-copy quarantine, an agreement-free trust score and title-linked evidence expansion, without NLI. On 700 fresh HotpotQA questions (study 6), v2 was more accurate than Vanilla RAG (+0.047 and +0.059 for two models) and than a cross-encoder reranker (+0.100 and +0.113). It gave fewer wrong answers (−0.083 and −0.107) and ran 3–4 times faster than Vanilla RAG. An abstention fallback raised the gain over Vanilla RAG to +0.067 and +0.091 (all Holm-corrected p ≤ 0.021). On 200 MuSiQue questions, v2 cut wrong answers (−0.155 and −0.140) but did not improve accuracy. In study 5, v2 blocked copy-based poisoning completely, but by abstaining, and it matched a source filter rather than beating it.

**Keywords:** retrieval-augmented generation; evidence selection; knowledge conflicts; corpus poisoning; abstention; pre-registration.

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

This paper reports such a test, and then uses what it found. We make five contributions:

1. **A diagnosis of why NLI-based conflict verification fails** (studies 1–4: three pre-registered studies on 900 questions and a pre-registered Monte Carlo study of 3,064,800 selections). The original CAS did not improve answers and did not resist poisoning. Four causes were measured directly:
   - the NLI model scores a poisoned copy of a passage as *agreeing* with its source, so the agreement bonus rewards the poison;
   - provenance, entered as one weight among four, cannot outweigh relevance;
   - the NLI detector catches under 10% of the conflicts that matter, at 20–35 s per question on CPU;
   - dropping one supporting passage costs more accuracy than keeping a distractor.
2. **CAS v2, a lightweight evidence selector built from that diagnosis** (Section 3.4). It has four steps, one per cause: a provenance gate, a quarantine for edited copies, a trust score without agreement, and title-linked evidence expansion. It has no NLI model, and it selects in about 3 ms per question. An optional abstention fallback asks again with every passage that passed the gate and the quarantine.
3. **Confirmation of CAS v2 in two pre-registered studies on fresh questions.**
   - Study 5 (350 HotpotQA questions) covers clean retrieval, the copy attack, and an attack written by a separate LLM.
   - Study 6 (700 HotpotQA and 200 MuSiQue questions) was powered from study 5's variance. Its runner would not start until the analysis plan had been committed to git.
   - On HotpotQA, v2 was more accurate than Vanilla RAG and a cross-encoder reranker, gave fewer wrong answers, and ran 3–4 times faster than Vanilla RAG.
4. **The limits of the method, stated in advance and confirmed.**
   - On MuSiQue's 2–4-hop questions, v2 cut wrong answers but did not raise accuracy.
   - Against copy attacks, it protects by abstaining.
   - When source metadata exist, it is no better than a source filter.
   - Against varied LLM-written poison, it gives no defence.
5. **A reproducible CPU-only evaluation.** All code, prompts, cached model outputs, analysis plans and figure data are released. Every figure is regenerated and checked against the result files by one script.

The original design failed, and we report its failure in full, because the failure was specific and measurable. The redesign shows that the failure was fixable where the diagnosis applied: evidence recall, cost and copy-based poisoning. It also shows where it was not: multi-hop chains that share no words with the question, and poison that is not a copy.

## 2. Related Work

**Retrieval-augmented generation.** RAG [1] and its many variants [2] condition generation on retrieved text. Later systems add self-critique of retrieved passages (Self-RAG [10]) or a lightweight retrieval evaluator that triggers corrective actions (CRAG [11]). These methods need trained components, and they are not evaluated here; Section 7 discusses this. Long and noisy contexts degrade LLM answers [3], [4], which motivates filtering before generation.

**Knowledge conflicts.** Longpre et al. [5] built entity-substitution conflicts between retrieved context and model memory. Chen et al. [6] studied conflicts among multiple retrieved sources. Xie et al. [7] showed that LLMs accept coherent counter-memory evidence. Xu et al. [8] survey the area. WikiContradict [14] collects real contradictions from Wikipedia, and RAGTruth [15] annotates hallucinations in RAG outputs. We use both.

**Corpus poisoning.** PoisonedRAG [9] showed that a few injected passages, written to be retrieved and to state a target answer, can control RAG output. Certified and aggregation-based defences have been proposed in response [12], [13]. Our study 2 follows the black-box PoisonedRAG design: several mutually consistent poison passages per question, which outnumber the true passage.

**NLI as a verifier.** Cross-encoder NLI models such as DeBERTa-v3 [16], trained on MultiNLI [17] and similar corpora, are widely used to check consistency between texts. Their behaviour on near-duplicate passages that differ in a single entity is central to our results (Section 6.3).

**Pre-registration.** Pre-registration fixes hypotheses and analyses before data collection, so that confirmatory and exploratory results can be told apart [18]. It is still rare in NLP evaluation. We use it to make negative results interpretable.

## 3. The CAS Framework

### 3.1 Pipeline

For a question *q* with retrieved passages *d₁…dₙ*, CAS proceeds in three steps (Fig. 1, top).

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

CAS v2 (`evaluation/cas_v2.py`; Fig. 1, bottom) was designed after studies 1–4, using only their data. It changes one thing for each measured cause of CAS's failures (Section 6). In study 5, the original design is called CAS v1: conflict-preserving CAS-NLI with θ = 0.99.

| Step | Rule | Cause addressed |
|---|---|---|
| Gate | When source metadata exist, passages with provenance < 0.5 are removed. | Provenance as a soft weight could not outweigh relevance |
| Quarantine | Two passages are *copies* when ≥ 80% of the shorter one's content words align, in order, with the other (κ = 0.8, at least 5 content words). A copy is *edited* when a run of 1–8 content words is changed, inserted or deleted inside it. Copies are grouped transitively. Every passage in a group that contains an edited copy is removed. | Near-copies "agreed" with their source, and the NLI graph caught under 10% of conflicts |
| Trust | 0.35·*r* + 0.25·*p* + 0.20·*c* ≥ τ = 0.45: v1's weights and τ, without the agreement term | Agreement rewarded the poison |
| Links | A remaining passage whose title (without a parenthetical) is named in the question or in a kept passage is kept too. | Dropping one supporting passage was costly |

v2 has no NLI graph and no conflict notes. The quarantine removes *both* sides of an edit dispute, because the passages alone cannot tell the original from the copy. Section 6.9 shows what that costs.

**Algorithm 1. CAS v2 evidence selection.**

```
Input:  question q; passages d_1..d_n with titles t_i (may be empty); provenance p_i (optional)
Output: the set S of passages given to the generator

1  P ← {1..n}
2  if provenance is available: P ← {i ∈ P : p_i ≥ 0.5}                     # gate
3  W_i ← content words of d_i, for i ∈ P
4  for each pair i < j in P:                                               # quarantine
5      align W_i and W_j (longest matching blocks)
6      if matched words ≥ 0.8 · min(|W_i|, |W_j|) and min(|W_i|, |W_j|) ≥ 5:
7          link i and j as copies (union-find)
8          if some non-matching run has 1..8 words: mark the pair as edited
9  P ← P minus every passage in a copy group that contains an edited pair
10 r_i ← BM25(q, d_i) / max_k BM25(q, d_k);  c_i ← 1 − rank_i / n;  p_i ← 0.5 if no provenance
11 K ← {i ∈ P : 0.35·r_i + 0.25·p_i + 0.20·c_i ≥ 0.45}                     # trust
12 L ← {i ∈ P \ K : title t_i (without parenthetical, ≥ 4 characters)
                    occurs as a phrase in q or in some d_k, k ∈ K}           # links
13 return S ← K ∪ L
```

**Cost.**
- With *n* passages of at most *m* content words, the quarantine compares every pair of passages, so the worst case is O(*n*²·*m*²). BM25 scoring and the link test are linear in the total text.
- With *n* ≤ 20, as in all our studies, selection took 2.7 ms per question on HotpotQA and 6.7 ms on MuSiQue, in single-threaded Python (study 6). No model inference is involved.
- By comparison, the NLI graph of CAS v1 scores n(n − 1) ordered pairs with a 184M-parameter cross-encoder, which took 20–35 s per question.

### 3.5 Abstention fallback

A filter that drops a needed passage leaves the generator unable to answer (Section 6.1), and small models then mostly abstain (Section 6.2). The fallback uses this signal:
1. Answer with v2's selection *S*.
2. If the model abstains, ask again with every passage that passed the gate and the quarantine (*P* after line 9 of Algorithm 1). Score that second answer.

The fallback costs a second generation only on abstentions. It never brings back a passage that the gate or the quarantine removed, so it keeps v2's protection against copies. It was added after study 5, on development data, and is tested in study 6 (Section 4.8).

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
| MuSiQue development check (study 6) | Pilot and development only | 200 (60 answered) | 20 |
| HotpotQA-6 (study 6) | Pre-registered powered test of CAS v2 | 700 | 10 |
| MuSiQue-6 (study 6) [21] | Pre-registered generalisation test of CAS v2 | 200 | 20 |

The HotpotQA samples of studies 1–3, 5 and 6 are drawn from the HotpotQA dev set. Each excludes every question used, attacked or attempted earlier, so the five samples do not overlap. The MuSiQue samples come from the answerable MuSiQue dev set. The study 6 sample excludes the development-check questions by id and by question text.

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

**Study 6 (clean retrieval, two datasets).**
- **HotpotQA:** 700 questions of every type; 10 passages each.
- **MuSiQue:** 200 answerable questions needing 2–4 hops, with 20 paragraphs each. Gold answers include the listed aliases.
- Both samples were drawn with seed 42 by `build_study6.py`. The runner called it only after the plan had been committed (Section 4.5).

**Source metadata.** Each poisoning study runs in two conditions:
- **Condition A:** no metadata; every passage has provenance 0.5.
- **Condition B:** noisy metadata. Trusted sources score U(0.55, 0.95), untrusted sources U(0.05, 0.45), and each passage's trusted/untrusted status is flipped with probability 0.2.

### 4.2 Methods compared

- **Vanilla RAG:** keeps all passages.
- **BM25 top-3.**
- **Reranker top-3** (studies 3, 5 and 6): `cross-encoder/ms-marco-MiniLM-L-6-v2` [22], [23].
- **Provenance + BM25 top-3** (condition B; "source filter"): a hard source filter, provenance ≥ 0.5, then BM25 top-3.
- **CAS:** w/o graph, keyword, NLI penalty, and NLI conflict-preserving. Conflict-preserving CAS is the primary method in studies 1–4.
- **CAS v2** (Section 3.4): the primary method in studies 5 and 6.
- **CAS v2 + fallback** (Section 3.5): a second arm in study 6.

### 4.3 Generation

- **Models:** Llama 3.2 3B and Gemma 3 4B in all studies, plus Llama 3.1 8B on the clean benchmark [24], [25]. Llama 3.1 8B also wrote the study 5 attack texts.
- **Runtime:** Ollama on CPU, temperature 0, seed 42, context window 8,192 tokens, at most 96 generated tokens.
- **Software:** Python 3.14.6, PyTorch 2.15.0 (CPU build), Sentence-Transformers 6.2.0 and Transformers 5.21.0. These were installed on 2026-10-29, before the first NLI prediction appears in the repository, and were not changed afterwards.
- **Ollama version:** the version was not stored with each cached response; this account comes from Ollama's server logs.
  - 0.40.12 served every answer generated on 2026-10-29 and 2026-10-30: the exploratory comparison and studies 1–2.
  - Ollama updated itself to 0.40.13 on 2026-11-03, before study 3's answers were generated. That version served studies 3, 5 and 6.
  - In study 6, the runner checked the version before every question and would have stopped on any other version; it never stopped.
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
  - study 5: 2026-11-04;
  - study 6: 2026-11-05.
- **What the git history independently confirms:**
  - Study 2: the plan was committed before its benchmark.
  - Study 3: the plan was committed before any output. The plan shares a commit with the 500-question sample, which a seeded script draws deterministically from the plan's own rules.
  - Study 5:
    - A first version of the plan, with hypotheses H1–H8, was committed before any study 5 data existed.
    - The final version adds H9, the attack-text rules (prompt, extraction, pilot) and the development results. It first appears in a later commit, together with the first 514 attacker outputs. That commit precedes the benchmark file, every selection and every answer.
    - The plan states that the final version was completed before the attack generation started.
  - Study 6:
    - The runner would not start until the plan was committed with the exact content on disk. It recorded that commit (`a5cc6153`, 2026-11-05 12:25:01 +09:00) in its output.
    - It drew the samples only after that point.
    - Earlier drafts of the plan are also in the history.
  - Studies 1 and 4: the plan first appears in the same commit as the benchmark (study 1) or the outputs (study 4). For these two, the timing rests on the plan's stated date.
- Each output file stores the SHA-256 of its plan. The stored hashes match the files:

| Study | Plan hash (prefix) |
|---|---|
| 1 | 47224db4 |
| 2 | a06a32f1 |
| 3 | c1eb4a89 |
| 4 | 2038df84 |
| 5 | c91b8c27 |
| 6 | 0f1fc82b |

- **Confirmatory families:**
  - studies 1–2: 20 tests each (2 models × 5 comparisons × 2 endpoints), five mechanical decisions;
  - study 3: 9 tests plus a non-inferiority margin of −0.05;
  - study 5: 20 tests plus a non-inferiority margin of −0.05 (Section 4.7);
  - study 6: a primary family of 8 tests, a secondary family of 12, and a non-inferiority margin of −0.05 (Section 4.8).
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

### 4.8 Study 6 design

Study 6 asks study 5's clean-retrieval questions again, with enough questions to answer them and on a second dataset. CAS v2 was frozen exactly as in study 5.

**Development before the plan.** All of it used data that study 6 does not use.
- **Pilot** (`dev_study6.py`). The datasets and comparisons were fixed before any method ran on MuSiQue. A pilot on 200 MuSiQue dev questions, 60 of them answered, then showed v2 losing evidence on multi-hop chains:
  - every supporting paragraph was kept for 43.5% of questions, against 91% on HotpotQA in study 5;
  - accuracy for Llama 3.2 was 0.133 vs 0.267 for Vanilla RAG.
- **Variants tried** (`dev_study6_variants.py`).
  - Not adopted: BM25 hop expansion, which kept most of the 20 paragraphs, and an iterative cross-encoder chain, which reached at most 51% full evidence.
  - Adopted as a second arm: the abstention fallback. On study 5's already-analysed clean questions it was +0.075 and +0.070 more accurate than Vanilla RAG (unadjusted).
  - The fallback was therefore chosen after seeing development results, and study 6 is its first test.

**Hypotheses** (per dataset; v2 against each comparator unless stated).

| | Hypothesis | Family |
|---|---|---|
| H1 | v2 is more accurate than Vanilla RAG | primary |
| H2 | v2 gives fewer wrong answers than Vanilla RAG | primary |
| H3 | v2 is non-inferior to Vanilla RAG in accuracy (95% CI lower bound above −0.05) | CI rule |
| H4 | v2 is more accurate than Reranker top-3 | secondary |
| H5 | v2 has higher evidence F1 than Reranker top-3 and than BM25 top-3 | secondary |
| H6 | v2 + fallback is more accurate than Vanilla RAG | secondary |

**Tests and decisions.**
- Holm correction within each family: 8 primary tests and 12 secondary tests.
- The decision rule is study 5's: every test of a hypothesis must be significant in the predicted direction, for both models.
- BM25 top-3 was evaluated for selection only, to bound CPU time.

**Sample size.**
- The SD of the per-question accuracy difference (v2 − Vanilla RAG) in study 5 was 0.389 for Llama 3.2 and 0.403 for Gemma 3.
- From these, n = 700 HotpotQA questions gives a power of 0.81–0.84 for a true difference of +0.055 at the strictest Holm threshold (α = 0.05/8).
- MuSiQue (n = 200) is a generalisation test.

**Predictions stated in the plan.**
- **HotpotQA:** H1, H3, H4, H5 and H6 supported; H2 uncertain for Gemma 3.
- **MuSiQue:** H1, H2, H3 and H5 not supported; H4 and H6 uncertain.

## 5. Results

### 5.1 Clean retrieval (exploratory)

**Evidence selection.**
- Conflict-preserving CAS with the cross-validated θ reached evidence F1 0.632 on HotpotQA (P 0.651, R 0.700; Fig. 2).
- It improved on Vanilla RAG by +0.299 [0.233, 0.360] (Holm p = 0.015, over 150 tests) and on CAS-keyword by +0.080 (Holm p = 0.024).
- Against BM25 top-3 it was more precise (+0.158, Holm p = 0.015), but the F1 difference was not significant (+0.040 [−0.009, 0.087]).
- It was indistinguishable from CAS w/o graph (−0.018, Holm p = 1.0). **The conflict graph therefore added nothing to selection.**

**Contradiction penalties.**
- The penalty removed 37 of 100 HotpotQA gold passages.
- It left an empty context for 19 of 65 questions with argmax NLI labels.
- It was significantly worse than the same pipeline without the graph (F1 −0.203, Holm p = 0.024).

**Answers.**
- After Holm correction over 210 tests, no CAS variant was significantly better than any baseline for any of the three LLMs (Fig. 3).
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

**Table 2. Pre-registered decisions** (primary method: conflict-preserving CAS in studies 1–3, CAS v2 in studies 5 and 6).

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
| 6 | HotpotQA | H1, accuracy vs Vanilla RAG | **supported** |
| 6 | HotpotQA | H2, fewer wrong answers than Vanilla RAG | **supported** |
| 6 | HotpotQA | H3, non-inferiority to Vanilla RAG | **supported** |
| 6 | HotpotQA | H4, accuracy vs Reranker top-3 | **supported** |
| 6 | HotpotQA | H5, evidence F1 vs Reranker and BM25 top-3 | **supported** |
| 6 | HotpotQA | H6, v2 + fallback accuracy vs Vanilla RAG | **supported** |
| 6 | MuSiQue | H1, accuracy vs Vanilla RAG | not supported (as predicted) |
| 6 | MuSiQue | H2, fewer wrong answers than Vanilla RAG | **supported** (predicted not to be) |
| 6 | MuSiQue | H3, non-inferiority to Vanilla RAG | not supported (Gemma 3 CI bound at the margin) |
| 6 | MuSiQue | H4, accuracy vs Reranker top-3 | not supported |
| 6 | MuSiQue | H5, evidence F1 vs Reranker and BM25 top-3 | not supported (below the reranker) |
| 6 | MuSiQue | H6, v2 + fallback accuracy vs Vanilla RAG | not supported (significant for Gemma 3 only) |

**The attack was weak.** Many substitutes were implausible, so poison adoption was near its floor even without filtering (Vanilla RAG: 3.5% for Llama, 6.5% for Gemma). That left little room to show a reduction.

**Accuracy.**
- Vanilla RAG was the most accurate method (Llama 0.505, Gemma 0.390).
- Conflict-preserving CAS was significantly less accurate than Vanilla RAG for Llama in condition A (−0.110 [−0.165, −0.055], Holm p = 0.012).

**Source filter.** In condition B, the filter gave Gemma 1.0% adoption, against 8.0% for conflict-preserving CAS (Holm p = 0.030).

**Preserve vs penalty (exploratory family).** Conflict preservation was more accurate than the penalty, by +0.060 to +0.065. The difference was significant in three of four cases (Holm p = 0.031–0.044).

All five decisions were "not supported" (Table 2; Fig. 4).

### 5.3 Study 2: a stronger attack (pre-registered)

**The attack was effective.** Without filtering, the models gave the poison answer in 28.0% (Llama) and 28.5% (Gemma) of questions, and accuracy fell to 0.230 and 0.220.

**Condition A (no metadata).** Conflict-preserving CAS did not differ from Vanilla RAG or BM25 top-3 on either endpoint (all Holm p = 1.0). A poison passage reached its context for every question.

**Condition B (noisy metadata).**
- The source filter cut adoption to 3.5% (Llama) and 4.0% (Gemma), with accuracy not significantly different from conflict-preserving CAS.
- Conflict-preserving CAS stayed at 22.5% and 28.0% (differences +0.190 and +0.240, both Holm p = 0.002).
- CAS w/o graph still selected a poison passage for 99.5% of questions.

**Penalty mode** gave the lowest CAS adoption (18.5–21.5%), with no accuracy gain and abstention of up to 20.5–28.5%.

All five decisions were "not supported" (Fig. 5).

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

**Parameters (E1).** Across 1,250 random weight-and-threshold settings, the comparisons with Vanilla RAG and with the source filter had the same sign as at the default setting in 100% of settings (Table 3; Fig. 7).
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

We evaluated 110 detector configurations (Fig. 8), spanning:
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

Five of the nine hypotheses were supported (Table 2; Table 4; Fig. 9). One of them, H7, had been predicted *not* to be supported.

**Table 4. Study 5: confirmatory tests** (CAS v2 minus comparator; 95% bootstrap CIs; p is Holm-corrected over the 20 tests; H4 is judged by its CI against the −0.05 margin). H1 compares selections, which do not depend on the model; its one value per comparator is in the Llama 3.2 column.

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

### 5.8 Study 6: a powered two-dataset test (pre-registered)

The study ran in full on Ollama 0.40.13. Its decisions are in Table 2 and its tests in Table 5; Fig. 10 shows the estimates.

**Table 5. Study 6: confirmatory tests** (method minus comparator; 95% bootstrap CIs; p is Holm-corrected within each family: primary, 8 tests; secondary, 12 tests). H5 compares selections, which do not depend on the model, so its one value per comparator is in the Llama 3.2 column.

| H | Data | Method vs comparator | Endpoint | Llama 3.2 | Gemma 3 | Decision |
|---|---|---|---|---|---|---|
| H1 | HotpotQA | v2 vs Vanilla RAG | Accuracy | +0.047 [+0.014, +0.080], p = 0.021 | +0.059 [+0.026, +0.091], p = 0.005 | **supported** |
| H2 | HotpotQA | v2 vs Vanilla RAG | Wrong answers | −0.083 [−0.116, −0.051], p < 0.001 | −0.107 [−0.139, −0.076], p < 0.001 | **supported** |
| H3 | HotpotQA | v2 vs Vanilla RAG | Accuracy (margin −0.05) | CI lower bound +0.014 | +0.026 | **supported** |
| H4 | HotpotQA | v2 vs Reranker top-3 | Accuracy | +0.113 [+0.080, +0.144], p = 0.001 | +0.100 [+0.067, +0.131], p = 0.001 | **supported** |
| H5 | HotpotQA | v2 vs Reranker top-3 | Evidence F1 | +0.089 [+0.069, +0.109], p = 0.001 | — | **supported** |
| H5 | HotpotQA | v2 vs BM25 top-3 | Evidence F1 | +0.126 [+0.109, +0.144], p = 0.001 | — | |
| H6 | HotpotQA | v2 + fallback vs Vanilla RAG | Accuracy | +0.067 [+0.036, +0.099], p = 0.001 | +0.091 [+0.061, +0.121], p = 0.001 | **supported** |
| H1 | MuSiQue | v2 vs Vanilla RAG | Accuracy | +0.015 [−0.035, +0.070], p = 1.0 | +0.010 [−0.050, +0.065], p = 1.0 | not supported |
| H2 | MuSiQue | v2 vs Vanilla RAG | Wrong answers | −0.155 [−0.225, −0.090], p < 0.001 | −0.140 [−0.215, −0.070], p = 0.003 | **supported** |
| H3 | MuSiQue | v2 vs Vanilla RAG | Accuracy (margin −0.05) | CI lower bound −0.035 | −0.050 | not supported |
| H4 | MuSiQue | v2 vs Reranker top-3 | Accuracy | +0.045 [−0.005, +0.095], p = 0.24 | +0.020 [−0.035, +0.075], p = 0.59 | not supported |
| H5 | MuSiQue | v2 vs Reranker top-3 | Evidence F1 | −0.062 [−0.097, −0.028], p = 0.002 | — | not supported |
| H5 | MuSiQue | v2 vs BM25 top-3 | Evidence F1 | +0.042 [+0.015, +0.067], p = 0.015 | — | |
| H6 | MuSiQue | v2 + fallback vs Vanilla RAG | Accuracy | +0.045 [0.000, +0.090], p = 0.24 | +0.070 [+0.025, +0.115], p = 0.026 | not supported |

**HotpotQA: every hypothesis supported, including H2, which was uncertain for Gemma 3.**
- **Evidence.** v2 reached evidence F1 0.719, against 0.630 for the reranker, 0.593 for BM25 top-3 and 0.340 for Vanilla RAG. It kept every supporting passage for 86.3% of questions, against 59.0% and 50.4% for the top-3 filters.
- **Accuracy.** v2 was the most accurate single-call method for both models:
  - Llama 3.2: 0.554, against 0.507 for Vanilla RAG and 0.441 for the reranker;
  - Gemma 3: 0.470, against 0.411 and 0.370.
- **Wrong answers.** They fell from 0.454 to 0.371 (Llama 3.2) and from 0.480 to 0.373 (Gemma 3). Abstention rose by 0.036 and 0.049 (exploratory; Holm p = 0.004 and 0.003).
- **The fallback.** It added a second call on 7.4% (Llama 3.2) and 15.7% (Gemma 3) of questions. It raised accuracy to 0.574 and 0.503, +0.020 and +0.033 over v2 (exploratory; Holm p = 0.003 and 0.002). It still gave fewer wrong answers than Vanilla RAG (0.400 and 0.417).
- **Study 5 replicated.** Study 5's non-significant accuracy gains (+0.065 and +0.055 at n = 200) replicated at n = 700 with similar sizes (+0.047 and +0.059). The H4 gain over the reranker repeated study 5's exploratory result (+0.105 and +0.095).

**MuSiQue: no accuracy gain, but far fewer wrong answers.**
- **Evidence.** v2 kept every supporting paragraph for 40.0% of questions, against 20.5% for the reranker and 8.0% for BM25 top-3. Its F1 (0.447) was below the reranker's (0.509), because it kept 6.3 of 20 paragraphs where the reranker kept 3.
- **Accuracy.** Accuracy was low for every method (0.12–0.23), and v2 did not differ from Vanilla RAG (+0.015 and +0.010).
- **Wrong answers.** v2 cut them sharply, from 0.595 to 0.440 (Llama 3.2) and from 0.500 to 0.360 (Gemma 3). The models abstained instead: abstention was 0.395 and 0.470, against 0.255 and 0.340. This was H2, predicted *not* to be supported. The prediction was wrong.
- **Non-inferiority (H3).** It failed for Gemma 3 by the narrowest margin: the CI lower bound was exactly −0.050, and the rule required a value above −0.05.
- **The fallback.** It raised accuracy to 0.195 and 0.230, significantly above Vanilla RAG for Gemma 3 only. It also brought wrong answers back to Vanilla RAG's level (0.600 and 0.525).
- **The pilot overestimated v2's harm.** In the 60-question pilot, v2 had been 0.133 less accurate than Vanilla RAG for Llama 3.2. In the study it was 0.015 more accurate. Vanilla RAG's own accuracy differed between the two samples (0.267 in the pilot, 0.150 in the study), so the pilot's estimate was mostly sampling noise.

**Cost.**
- **Selection:** v2 selected in 2.7 ms (HotpotQA) and 6.7 ms (MuSiQue) per question, against 0.15 s and 0.30 s for the reranker.
- **Prompt length:** v2's prompts were 34–35% of Vanilla RAG's on HotpotQA (487–488 vs 1,402–1,429 tokens), and 36–37% on MuSiQue (898–911 vs 2,464–2,542).
- **Generation time:**
  - HotpotQA: 2.6 s (Llama 3.2) and 4.1 s (Gemma 3), against 10.0 s and 12.2 s for Vanilla RAG;
  - MuSiQue: 6.0 s and 7.6 s, against 19.9 s and 21.5 s.
- **The fallback's cost** includes its second call: 3.3 s and 5.9 s on HotpotQA, and 13.3 s and 17.2 s on MuSiQue.

### 5.9 What each step of CAS v2 contributes (exploratory ablation)

Each step of v2 was switched off in turn on the study 5 and study 6 data (`evaluation/run_v2_ablation.py`). This is a selection-level analysis with no model calls, chosen after the studies were analysed. The full-v2 selections it recomputes match the cached ones exactly.

**Table 6. Ablation of CAS v2** (selection level; exploratory).

| Data | Variant | Evidence F1 | All supporting passages kept | Passages kept | Copy attack: poison in prompt | Copy attack: true passage kept |
|---|---|---|---|---|---|---|
| HotpotQA (study 6, 700) | CAS v2 | 0.719 | 86.3% | 3.41 | | |
| | without title links | 0.627 | 42.3% | 2.61 | | |
| | without quarantine | 0.723 | 87.7% | 3.47 | | |
| | trust threshold only | 0.631 | 42.9% | 2.65 | | |
| MuSiQue (study 6, 200) | CAS v2 | 0.447 | 40.0% | 6.30 | | |
| | without title links | 0.408 | 17.0% | 5.09 | | |
| | without quarantine | 0.442 | 40.0% | 6.43 | | |
| | trust threshold only | 0.403 | 17.0% | 5.18 | | |
| Copy attack, no metadata (study 5, 150) | CAS v2 | | | | 0.7% | 0% |
| | without title links | | | | 0% | 0% |
| | without quarantine | | | | 100% | 96.7% |
| | trust threshold only | | | | 100% | 22.0% |

- **The title links supply the evidence recall.** Without them, full evidence falls by about half on both datasets (86.3% → 42.3% and 40.0% → 17.0%).
- **The quarantine supplies the copy defence.** Without it, poison reaches the prompt for every copy-attack question. On clean data it costs little (86.3% vs 87.7% full evidence on HotpotQA).
- **The two interact under attack.** The links pull in poisoned copies that keep the original's title (2.93 poison passages per question without the quarantine, against 1.42 for the trust threshold alone). The quarantine removes them. The steps are complementary, not redundant.

## 6. Discussion

The results raise six questions:
- Why did better evidence not give better answers (Sections 6.1–6.2)?
- Why does CAS not stop poisoning (Section 6.3)?
- Why do conflict notes not change answers (Section 6.4)?
- What does the conflict graph cost (Section 6.5)?
- Did fixing the diagnosed causes work (Section 6.9)?
- Where does the fixed method work, and where does it stop (Section 6.10)?

Sections 6.1, 6.2, 6.4 and 6.5 use exploratory analyses of the cached study 3 outputs, chosen after the results were known (`evaluation/run_discussion_analyses.py`, output `results/discussion_analyses.json`). They were not pre-registered. Section 6.9 likewise uses a post hoc analysis of study 5's selections (`evaluation/run_study5_mechanism.py`, output `results/study5_mechanism.json`), and Section 6.10 interprets study 6 with the exploratory ablation of Section 5.9. These analyses help explain the confirmatory results but do not replace them, and each would need its own confirmatory test.

### 6.1 Why better evidence did not give better answers

In study 3, CAS doubled evidence F1 relative to Vanilla RAG, yet answered less accurately. This is not specific to CAS. Every filter, including BM25 top-3 and the reranker, was less accurate than Vanilla RAG for both models (Section 5.4).

**Filtering helps when it keeps the evidence, and hurts when it drops it.** We split the 500 questions by whether a filter kept every supporting passage. Within each group, we compared the filter with Vanilla RAG on the same questions (Table 7).
- **When a filter kept every supporting passage,** it was *more* accurate than Vanilla RAG in all ten filter–model combinations (+0.039 to +0.062). The 95% CI excluded zero in eight of them. After Holm correction over the 20 stratified tests, only one remained significant: BM25 top-3 with Llama 3.2 (Holm p = 0.044).
- **When a filter dropped a supporting passage,** it was much less accurate (−0.129 to −0.258; all Holm p ≤ 0.003).
- **Most of CAS's losses were in the second group.** Take the questions that Vanilla RAG answered correctly and CAS did not. CAS had dropped a supporting passage on 48 of 61 of them (Llama 3.2) and on 58 of 70 (Gemma 3).

**Table 7. Accuracy by whether the filter kept every supporting passage** (study 3; exploratory; each filter is paired with Vanilla RAG on the same questions).

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

Accuracy counts an abstention and a wrong answer the same way. Table 8 separates them.

**Table 8. Answer outcomes on the 500 clean questions of study 3** (share of questions; exploratory).

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
2. **Report correct, wrong and abstained answers separately.** Accuracy alone hides the trade shown in Table 8.
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

### 6.10 What study 6 adds: where the method works, and where it stops

**1. On HotpotQA, the evidence-recall diagnosis is now confirmed at the answer level.**
- Section 6.1 estimated, from study 3, that a filter would need to keep every supporting passage for about 74–80% of questions to match Vanilla RAG's accuracy.
- v2 kept them for 86.3% in study 6, and it was significantly *more* accurate than Vanilla RAG.
- The reranker kept them for 59.0%, and it was less accurate than Vanilla RAG.
- The ablation (Section 5.9) attributes most of the recall to the title links.
- So the lever that the diagnosis identified is the one that moved the answers.

**2. v2 dominates Vanilla RAG on both datasets, though only partly significantly.**
- In point estimates, v2 gave more correct answers *and* fewer wrong answers than Vanilla RAG, for both models on both datasets (Section 5.8).
- It therefore scores higher under any cost of a wrong answer: +1 for correct, 0 for an abstention, −*c* for wrong, for any *c* ≥ 0. This is unlike CAS v1 in study 3, which beat Vanilla RAG only when *c* > 0.44–0.76 (Section 6.2).
- Among the questions it answered, v2 was right more often than Vanilla RAG in all four cases. These figures are computed from the shares in Section 5.8:
  - HotpotQA: 0.599 vs 0.527 (Llama 3.2) and 0.558 vs 0.462 (Gemma 3);
  - MuSiQue: 0.273 vs 0.201 and 0.321 vs 0.242.
- On MuSiQue, only the reduction in wrong answers is statistically supported. The accuracy differences are not.

**3. On MuSiQue, the lever does not reach far enough.**
- v2 kept the full evidence chain for 40.0% of questions, far below the level at which filtering pays off in accuracy.
- The later hops of a MuSiQue question are linked to it through an intermediate answer, not through shared words or a named title.
- What v2 does on MuSiQue is what CAS v1 did on HotpotQA (Section 6.2): when the evidence is incomplete, the model abstains instead of guessing.
- Two cheap remedies did not close this gap in development (Section 4.8). It probably needs a retriever that follows bridge entities, such as one trained for multi-hop retrieval or one that decomposes the question. We did not test either.

**4. The fallback is a dial, not a free gain.**
- On HotpotQA, asking again after an abstention added accuracy (+0.020 and +0.033 over v2) and kept wrong answers below Vanilla RAG's.
- On MuSiQue, it recovered more accuracy, but it also brought wrong answers back to Vanilla RAG's level. When v2's abstentions were asked again with the wider pool, only 7.6% (Llama 3.2) and 12.8% (Gemma 3) became correct answers, while 40.5% and 35.1% became wrong ones (computed from the shares in Section 5.8).
- So v2 alone suits settings where a wrong answer is costly, and v2 + fallback suits settings where coverage matters more.

**5. A small pilot misled the predictions, and pre-registration made that visible.**
- The 60-question pilot made v2 look clearly harmful on MuSiQue for Llama 3.2 (−0.133). We therefore predicted that H2 would fail.
- On 200 questions, v2 was level in accuracy and significantly reduced wrong answers.
- Because the prediction was written down, the error shows. Readers can see that our expectations, not the data, were wrong.
- Pilot estimates at n = 60 should inform the plan's direction, not its predictions of effect size.

## 7. Threats to Validity

**Internal validity.**
- θ was chosen on the 65 clean questions and frozen. The weights and τ are untuned defaults; study 4 shows that the conclusions do not depend on them.
- In study 2, the substitute rules were amended once, before any method ran (documented in the plan).
- Some substitutes remain semantically mismatched (for example, a person's name replacing a place). This makes the attack easier to resist, not harder.
- **CAS v2 was designed after studies 1–4, on their data, including the copy attack.**
  - H5 therefore tests whether v2 generalises to new questions, not to a new attack.
  - The written attack is the only out-of-family test, and the one test where v2's gain turned out to be incidental.
- **The study 5 mechanism analysis (Section 6.9) is post hoc.**
- **v2's settings were chosen on HotpotQA data** (studies 1–3). Study 6's HotpotQA result is therefore a replication on fresh questions from the same distribution. MuSiQue is the out-of-distribution test, and there v2 did not raise accuracy.
- **The abstention fallback was chosen after the study 6 pilot**, on development data: study 5's clean questions and the MuSiQue pilot. Study 6 is its only test so far.
- **The Ollama runtime changed between studies.** It moved from 0.40.12 to 0.40.13 between study 2 and study 3 (Section 4.3), with the same model weights. Comparisons within a study are unaffected, because every method in a study ran on the same version. Comparisons of absolute accuracy across studies could be affected.

**Construct validity.**
- Accuracy and adoption are string-match metrics. Human validation is in progress (**[HV]**). The same applies to the "wrong" category in Table 8: a correct paraphrase that does not contain the gold string counts as wrong.
- Silent exposure (study 4) is a selection-level proxy. E4 suggests that, for small models, a conflict note does not change adoption, so the proxy may overstate the value of marking.
- **The written attack (study 5) uses one attacker model and one prompt.**
  - The prompt adds two sentences to PoisonedRAG's.
  - 17% of the sampled questions were skipped because the attacker failed, so the attacked set is conditional on the attacker's success.
  - An attacker that diversifies its texts would remove v2's incidental gain (Section 6.9).
  - The attacker, Llama 3.1 8B, belongs to the same model family as one answering model, Llama 3.2 3B.
- **On MuSiQue, accuracy was low for every method (0.12–0.23).** The answer-level comparisons there are close to a floor, and they may not carry over to models that reason better over long contexts.

**External validity.**
- All generators are 3–8B open models run on CPU. Larger models may use conflict notes differently.
- The poisoning studies use HotpotQA only.
- The provenance scores are synthetic, because no benchmark we could access carries real source-reliability labels.
- Self-RAG [10] and CRAG [11] were not re-implemented.
- v2's link rule uses the passage titles that HotpotQA and MuSiQue provide. Both are built from Wikipedia. A production retriever may not supply titles in the same form.
- In study 5, CAS v1 was not run under the written attack or with metadata, to bound CPU time.

**Statistical conclusion validity.**
- The clean comparison has n = 65 and is exploratory; study 3 replicates it at n = 500.
- The confirmatory families are Holm-corrected.
- The Monte Carlo study reports question-level CIs only. Its run count reduces simulation noise, not sampling error.
- Study 5's clean set is small (n = 200). Its accuracy differences have 95% CIs about ±0.05 wide, so differences of 0.055–0.065 did not survive Holm correction over 20 tests. The plan anticipated this for Gemma 3 (H3, H9).
- Study 6's MuSiQue sample is small (n = 200). There, H3 failed for Gemma 3 with a CI lower bound exactly at the −0.05 margin, so a slightly different sample could have reversed this decision.
- The analyses in Sections 5.9, 6.1, 6.2, 6.4, 6.5, 6.9 and 6.10 were chosen after the results were known. Their tests are Holm-corrected within each analysis, but the set of analyses was not fixed in advance. The groups in Table 7 are defined by each method's own selection, so they are not randomly assigned.

**Pre-registration.**
- The plans are hash-stamped in the outputs but were not deposited in a public registry.
- Git history independently confirms that the plan came first for studies 2, 3 and 6, and partly for study 5. For study 6, the runner itself refused to start until the plan was committed.
  - For studies 1 and 4, the plan first appears in the same commit as the benchmark or the outputs.
  - For study 5, git confirms a first version of the plan before any data. It confirms the final version before the benchmark and every answer, but not before the first attacker outputs (Section 4.5).
- Future studies should be registered publicly before data generation.

## 8. Conclusion

We asked whether conflict-aware verification before generation helps retrieval-augmented generation. We answered it in six pre-registered studies, with the analysis fixed before the data existed.

**The original design did not work, for four measurable reasons.**
- Conflict-preserving CAS selected more precise evidence than Vanilla RAG and cut prompts to about 28% of their length. But it did not improve answers, either in the exploratory comparison or at n = 500 (study 3).
- In poisoning studies 1 and 2, it did not reduce poison adoption (differences −0.055 to +0.030, all Holm p = 1.0). A plain source filter did.
- A Monte Carlo study of more than three million selections, and the exploratory analyses, traced these results to four causes. They are listed in the table below.

**A redesign built from those reasons works where the diagnosis applies.** CAS v2 changes one thing per cause and needs no NLI model.

| Cause (studies 1–4) | Fix in CAS v2 | Pre-registered result (studies 5–6) |
|---|---|---|
| Dropping a supporting passage costs more than keeping a distractor | Keep passages linked by title | All supporting passages kept for 86–91% of HotpotQA questions (v1: 48.5%). Accuracy above Vanilla RAG (+0.047, +0.059) and the reranker (+0.100, +0.113) at n = 700 |
| The NLI detector catches fewer than 10% of conflicts, at 20–35 s per question | No NLI graph | Selection in 3–7 ms. 2.8–3.8 times faster than Vanilla RAG end to end, with prompts about a third as long |
| The agreement signal rewards poisoned near-duplicates | No agreement term; edited copies quarantined | Copy attack blocked (0% adoption, against 27–34% for Vanilla RAG), but by abstaining: the true passage is quarantined too |
| Provenance, as a soft weight, cannot outweigh relevance | Provenance as a gate | Equal to a plain source filter; no gain over it |

**Across both datasets, v2 gave fewer wrong answers than Vanilla RAG** (−0.083 to −0.155, all Holm p ≤ 0.003). In point estimates, it also gave more correct ones. An abstention fallback adds accuracy on HotpotQA (+0.067 and +0.091 over Vanilla RAG), at some cost in wrong answers.

**The method has clear limits, and each was tested.**
- On MuSiQue's 2–4-hop questions, v2 kept the full evidence chain for only 40% of questions and did not raise accuracy.
- It cannot tell an original passage from its edited copy, so it protects against copies by abstaining.
- It has no defence against varied poison that is not a copy.

Closing these gaps will need retrieval that follows bridge entities across hops, and provenance signals that current benchmarks do not carry. Every plan, cached model output and per-question result is released, so each claim above can be checked.

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
| Fig. 1 | CAS v1 and CAS v2 pipelines | `fig00_pipeline` |
| Fig. 2 | Clean evidence selection; paired differences | `fig01_precision_recall_tradeoff`, `fig02_paired_differences` |
| Fig. 3 | LLM answer quality and cost | `fig08_llm_answer_quality`, `fig10_llm_cost_tradeoff` |
| Fig. 4 | Study 1 | `fig12_poison_study1` |
| Fig. 5 | Study 2 | `fig13_poison_study2` |
| Fig. 6 | Study 3 | `fig16_clean500` |
| Fig. 7 | Monte Carlo study | `fig15_montecarlo` |
| Fig. 8 | Detector accuracy vs false alarms vs cost | `fig14_detector_tradeoff` |
| Fig. 9 | Study 5: CAS v2 on fresh questions | `fig17_study5` |
| Fig. 10 | Study 6: CAS v2 on 700 HotpotQA and 200 MuSiQue questions | `fig18_study6` |

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
