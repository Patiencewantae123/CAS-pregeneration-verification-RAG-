import sys
import os
import json
import streamlit as st
import pandas as pd

# Add python directory to system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.document import Document
from cas_engine import CASEngine
from services.nli_verifier import NLIVerifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")

# Page Configuration
st.set_page_config(
    page_title="CAS Verification Dashboard",
    page_icon="🛡️",
    layout="wide"
)


@st.cache_resource(show_spinner="Loading DeBERTa-v3 NLI cross-encoder...")
def load_nli_verifier():
    from services.nli_model_verifier import NLIModelVerifier
    verifier = NLIModelVerifier()
    verifier._load()
    return verifier


@st.cache_data
def load_json(name):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# Header Section
st.title("🛡️ Conflict-Aware Synthesis (CAS) Dashboard")
st.markdown("### *Pre-Generation Verification Layer for RAG Architectures*")
st.divider()

# Sidebar Controls
st.sidebar.header("🎛️ Pipeline Controls")

query = st.sidebar.text_input(
    "Target Query",
    value="Is Drug-X approved for Phase 3 trials?"
)

verifier_name = st.sidebar.radio(
    "Contradiction verifier",
    ["DeBERTa-v3 NLI model", "Keyword heuristic (original)"],
    help="The NLI model (cross-encoder/nli-deberta-v3-base) runs on CPU; the keyword heuristic "
         "only matches approved/rejected-style word pairs."
)

synthesis_mode = st.sidebar.radio(
    "Conflict handling",
    ["Conflict-preserving", "Contradiction penalty"],
    help="Conflict-preserving keeps both sides of an on-topic contradiction and labels it for the LLM. "
         "Contradiction penalty subtracts 0.5 x contradiction confidence from each document's weight."
)

trust_threshold = st.sidebar.slider(
    "Trust Threshold (τ)",
    min_value=0.10,
    max_value=0.80,
    value=0.45,
    step=0.05
)

st.sidebar.markdown("---")
st.sidebar.info("Adjust the controls to see real-time filtering of conflicting context nodes.")

# Load Test Dataset
docs = [
    Document("Doc_01", "The FDA approved drug-X for Phase 3 clinical trials in early 2024.", "PeerReviewed_Journal", 0.92, 0.90, 0.95),
    Document("Doc_02", "Regulators rejected drug-X for Phase 3 trials due to severe health concerns.", "Unverified_Blog", 0.65, 0.75, 0.25),
    Document("Doc_03", "Phase 3 clinical trial for drug-X was approved by health regulatory bodies.", "Medical_News_Outlet", 0.88, 0.85, 0.80),
    Document("Doc_04", "Drug-X is an experimental therapeutic compound tested for hypertension.", "Pharma_DB", 0.95, 0.92, 0.90)
]

# Execute CAS Engine
verifier = load_nli_verifier() if verifier_name.startswith("DeBERTa") else NLIVerifier()
engine = CASEngine(trust_threshold=trust_threshold, verifier=verifier,
                   preserve_conflicts=synthesis_mode == "Conflict-preserving")
results = engine.execute_pipeline(query, docs)

# Layout Columns for Overview Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Retrieved Passages", len(docs))
retained_count = sum(1 for v in results["verified_documents"] if not v.is_filtered)
col2.metric("Retained Passages", retained_count)
col3.metric("Rejected Passages", len(docs) - retained_count)
contradictions = sum(1 for e in results["graph"].edges if e.relation.value == "Contradiction")
col4.metric("Contradictions Detected", contradictions)

st.divider()

# Main Interface Tabs
tab1, tab2, tab3 = st.tabs(["📊 Adjudication Table", "📜 Reconciled Context (C_verified)", "📈 Measured Benchmark Results"])

with tab1:
    st.subheader("1. NLI Adjudication & Evidence Filtering")

    table_data = []
    for v in results["verified_documents"]:
        table_data.append({
            "Doc": v.doc.id,
            "Source": v.doc.source,
            "Content Passage": v.doc.content,
            "Trust Score": round(v.trust_score, 2),
            "Conflict Penalty": f"-{v.conflict_penalty:.2f}",
            "Net Weight": round(v.final_weight, 2),
            "Status": "✅ RETAINED" if not v.is_filtered else "❌ REJECTED"
        })

    df = pd.DataFrame(table_data)

    def color_status(val):
        if "RETAINED" in str(val):
            return "background-color: #064e3b; color: #86efac; font-weight: bold;"
        return "background-color: #7f1d1d; color: #fca5a5; font-weight: bold;"

    st.dataframe(
        df.style.map(color_status, subset=["Status"]),
        width="stretch",
        hide_index=True
    )

    st.markdown("**Pairwise NLI relations (conflict graph edges)**")
    st.dataframe(pd.DataFrame([{
        "Pair": f"{e.source_doc_id} ↔ {e.target_doc_id}",
        "Relation": e.relation.value,
        "Confidence": round(e.confidence, 3)
    } for e in results["graph"].edges]), width="stretch", hide_index=True)

with tab2:
    st.subheader("2. Reconciled Context ($C_{verified}$)")
    st.caption("Sanitized context passed directly to the LLM generator:")
    st.code(results["c_verified"], language="markdown")

with tab3:
    st.subheader("3. Measured Benchmark Results")
    st.caption("Read from evaluation/results/, produced by evaluation/run_evaluation.py and "
               "evaluation/run_llm_generation.py. Means with 95% bootstrap confidence intervals.")

    evaluation = load_json("results.json")
    if evaluation is None:
        st.info("No results yet. Run `py evaluation/run_evaluation.py` from the project root.")
    else:
        rows = {}
        for r in evaluation["evidence_selection"]:
            if r["metric"] in ("precision", "recall", "f1"):
                key = (r["dataset"], r["method"])
                rows.setdefault(key, {"Dataset": r["dataset"], "Method": r["method"]})
                rows[key][r["metric"].capitalize() if r["metric"] != "f1" else "F1"] = (
                    f"{r['mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")
        st.markdown("**Evidence selection** (HotpotQA n = 50, MuSiQue n = 15)")
        st.dataframe(pd.DataFrame(rows.values()), width="stretch", hide_index=True)

    llm = load_json("llm_summary.json")
    if llm is not None:
        rows = {}
        for r in llm["qa"]:
            if r["metric"] in ("accuracy", "f1"):
                key = (r["dataset"], r["model"], r["method"])
                rows.setdefault(key, {"Dataset": r["dataset"], "Model": r["model"], "Method": r["method"]})
                rows[key]["Answer accuracy" if r["metric"] == "accuracy" else "Token F1"] = (
                    f"{r['mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")
        st.markdown("**End-to-end answers from local LLMs**")
        st.dataframe(pd.DataFrame(rows.values()), width="stretch", hide_index=True)
