# CAS-PreGeneration-Verification-RAG
<img width="1904" height="953" alt="image" src="https://github.com/user-attachments/assets/b5e26cd0-bb94-40d8-b9b4-3a7051aa900f" />

### Conflict-Aware Synthesis for Reliable Retrieval-Augmented Generation

## 🎯 What Problem Does This Project Solve?

Traditional Retrieval-Augmented Generation (RAG) systems retrieve documents relevant to a user's question and provide them to a Large Language Model (LLM) to generate an answer. However, the retrieved documents may contain **contradictory information, unreliable sources, or conflicting facts**.

For example, when a user asks, *"What is the recommended treatment for a particular condition?"*, one retrieved document may provide outdated information while another provides updated guidance. A conventional RAG system may combine these sources and generate a misleading or contradictory answer.

**CAS-PreGeneration-Verification-RAG addresses this problem by introducing a verification layer before answer generation.** It analyzes retrieved evidence, detects contradictions, evaluates source reliability, and prioritizes consistent, relevant evidence before passing the refined context to the LLM.

## 🔍 The Specific Problems It Addresses

1. **Contradictory evidence:** Identifies when retrieved documents disagree with one another.
2. **Unreliable information:** Evaluates evidence using factors such as source reliability and agreement between documents.
3. **Hallucinations:** Aims to reduce unsupported statements by providing the LLM with better-verified evidence.
4. **Inconsistent answers:** Reduces the risk of generating answers that combine incompatible facts.
5. **Poor evidence selection:** Ranks retrieved passages so that more relevant and trustworthy evidence can receive greater priority.

## ⚙️ How Does It Solve These Problems?

1. **User Query:** Receives a question from the user.
2. **Document Retrieval:** Retrieves relevant documents from a knowledge source.
3. **Conflict Detection:** Uses Natural Language Inference (NLI) to identify potential contradictions between evidence passages.
4. **Conflict Graph Construction:** Represents relationships between passages, including supporting and contradicting evidence.
5. **Trust-Aware Evidence Ranking:** Evaluates evidence based on relevance, agreement, and source reliability.
6. **Conflict-Aware Synthesis:** Constructs a refined context that prioritizes reliable evidence and handles detected conflicts.
7. **LLM Generation:** Sends the refined context to the LLM to generate an evidence-grounded response.

## 🧪 Research Objective

The objective is to investigate whether verifying and synthesizing retrieved evidence before generation can reduce hallucinations and contradictions while improving factual consistency and reliability compared with conventional RAG and other relevant baselines.

## 💡 Expected Contribution

The proposed framework introduces a pre-generation verification approach that integrates contradiction detection, evidence relationship modeling, trust-aware ranking, and conflict-aware synthesis into the RAG pipeline.

**Research Focus:** Trustworthy AI · LLM Reliability · Retrieval-Augmented Generation · Evidence Verification

## 🔁 Reproducing the paper

Every result in `paper/manuscript.md` comes from the scripts in `evaluation/`; every figure is rebuilt and checked against the result files by `journal_figures/make_all.py`. The step-by-step commands are in [`journal_figures/README.md`](journal_figures/README.md#reproduce).

**Software** (the versions that produced the results):

| Component | Version |
|---|---|
| Python | 3.14.6 |
| PyTorch | 2.15.0 (CPU build) |
| Sentence-Transformers | 6.2.0 |
| Transformers | 5.21.0 |
| numpy / pandas / pyarrow / matplotlib | 2.5.1 / 3.0.3 / 24.0.0 / 3.11.0 |
| Ollama | 0.40.12 for the exploratory answers and studies 1–2; 0.40.13 for studies 3, 5 and 6 (an automatic update on 2026-11-03, between studies 2 and 3) |

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
ollama pull llama3.2 && ollama pull gemma3 && ollama pull llama3.1:8b
ollama list    # the IDs should match those below; a newer pull may bring different weights
```

**Models** (Ollama IDs, unchanged across studies): Llama 3.2 3B `a80c4f17acd5`, Gemma 3 4B `a2af6cc3eb7f`, Llama 3.1 8B `46e0c10c039e`. The NLI model is `cross-encoder/nli-deberta-v3-base` and the reranker `cross-encoder/ms-marco-MiniLM-L-6-v2`, both from the Hugging Face Hub.

**Hardware:** an Intel CPU with 15 GB RAM and no GPU. Model answers are cached in `evaluation/results/llm_generations.jsonl`, so the analyses re-run without Ollama.

Ollama's Windows app installs a downloaded update when it next starts. To reproduce a study on one runtime version, do not restart Ollama during a run; `evaluation/run_study6.py` checks the version before every question and stops if it changes.
