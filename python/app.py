import sys
import os
import streamlit as st
import pandas as pd

# Add python directory to system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.document import Document
from cas_engine import CASEngine

# Page Configuration
st.set_page_config(
    page_title="CAS Verification Dashboard",
    page_icon="🛡️",
    layout="wide"
)

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

trust_threshold = st.sidebar.slider(
    "Trust Threshold (τ)", 
    min_value=0.10, 
    max_value=0.80, 
    value=0.35, 
    step=0.05
)

st.sidebar.markdown("---")
st.sidebar.info("Adjust the threshold slider to see real-time filtering of conflicting context nodes.")

# Load Test Dataset
docs = [
    Document("Doc_01", "The FDA approved drug-X for Phase 3 clinical trials in early 2024.", "PeerReviewed_Journal", 0.92, 0.90, 0.95),
    Document("Doc_02", "Regulators rejected drug-X for Phase 3 trials due to severe health concerns.", "Unverified_Blog", 0.65, 0.75, 0.25),
    Document("Doc_03", "Phase 3 clinical trial for drug-X was approved by health regulatory bodies.", "Medical_News_Outlet", 0.88, 0.85, 0.80),
    Document("Doc_04", "Drug-X is an experimental therapeutic compound tested for hypertension.", "Pharma_DB", 0.95, 0.92, 0.90)
]

# Execute CAS Engine
engine = CASEngine(trust_threshold=trust_threshold)
results = engine.execute_pipeline(query, docs)

# Layout Columns for Overview Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Retrieved Passages", len(docs))
retained_count = sum(1 for v in results["verified_documents"] if not v.is_filtered)
col2.metric("Retained Passages", retained_count)
col3.metric("Rejected Passages", len(docs) - retained_count)
col4.metric("CAS CPI Score", "88.48", delta="+6.00 vs CRAG")

st.divider()

# Main Interface Tabs
tab1, tab2, tab3 = st.tabs(["📊 Adjudication Table", "📜 Reconciled Context (C_verified)", "📈 Benchmark CPI Metrics"])

with tab1:
    st.subheader("1. NLI Adjudication & Evidence Filtering")
    
    table_data = []
    for v in results["verified_documents"]:
        table_data.append({
            "Doc ID": v.doc.id,
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
        use_container_width=True,
        hide_index=True
    )

with tab2:
    st.subheader("2. Reconciled Context ($C_{verified}$)")
    st.caption("Sanitized context passed directly to the LLM generator:")
    st.code(results["c_verified"], language="markdown")

with tab3:
    st.subheader("3. Benchmark Evaluation (Table 2 Comparison)")
    
    benchmark_df = pd.DataFrame([
        {"Framework": "Baseline RAG", "Accuracy (%)": 72.1, "Hallucination (%)": 21.4, "Contradiction (%)": 18.9, "Consistency (%)": 70.3, "CPI Score": 75.48},
        {"Framework": "Self-RAG", "Accuracy (%)": 76.8, "Hallucination (%)": 16.2, "Contradiction (%)": 14.1, "Consistency (%)": 77.5, "CPI Score": 81.00},
        {"Framework": "CRAG", "Accuracy (%)": 78.4, "Hallucination (%)": 14.8, "Contradiction (%)": 12.9, "Consistency (%)": 79.2, "CPI Score": 82.48},
        {"Framework": "CAS (Proposed)", "Accuracy (%)": 82.7, "Hallucination (%)": 9.5, "Contradiction (%)": 6.1, "Consistency (%)": 86.8, "CPI Score": 88.48},
    ])
    
    st.dataframe(benchmark_df, use_container_width=True, hide_index=True)
