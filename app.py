import os
import tempfile
import numpy as np
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

st.set_page_config(
    page_title="casv  RAG Verification Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .main { padding: 1rem 2rem; }
    .stChatInput { position: fixed; bottom: 20px; }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
    .badge-retained {
        background-color: #d4edda;
        color: #155724;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .badge-rejected {
        background-color: #f8d7da;
        color: #721c24;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    </style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "bm25_retriever" not in st.session_state:
    st.session_state.bm25_retriever = None
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []

@st.cache_resource(show_spinner="Loading NLI Cross-Encoder Verifier...")
def load_verification_model():
    return CrossEncoder('cross-encoder/nli-deberta-v3-base')

@st.cache_resource(show_spinner="Loading Embedding Model...")
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

verifier = load_verification_model()
embeddings = load_embedding_model()

with st.sidebar:
    st.header("⚙️ Document Hub")
    st.markdown("Upload multiple PDF or TXT files to build your verified knowledge base.")
    
    uploaded_files = st.file_uploader(
        "Upload Documents", 
        type=["pdf", "txt"], 
        accept_multiple_files=True,
        help="Select one or more files simultaneously."
    )
    
    chunk_size = st.slider("Chunk Size", min_value=200, max_value=1000, value=500, step=50)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=200, value=50, step=10)

    if st.button("🚀 Process & Index Documents", use_container_width=True):
        if not uploaded_files:
            st.error("Please select at least one file before processing.")
        else:
            all_docs = []
            with st.spinner(f"Processing {len(uploaded_files)} file(s)..."):
                for uploaded_file in uploaded_files:
                    file_ext = uploaded_file.name.split(".")[-1].lower()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name

                    if file_ext == "pdf":
                        loader = PyPDFLoader(tmp_path)
                    else:
                        loader = TextLoader(tmp_path, encoding="utf-8")
                    
                    docs = loader.load()
                    for d in docs:
                        d.metadata["source"] = uploaded_file.name
                    all_docs.extend(docs)

                splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                chunks = splitter.split_documents(all_docs)
                
                st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
                st.session_state.bm25_retriever = BM25Retriever.from_documents(chunks)
                st.session_state.bm25_retriever.k = 5
                st.session_state.all_chunks = chunks
                
                st.success(f"Indexed {len(chunks)} chunks across {len(uploaded_files)} document(s)!")

    st.markdown("---")
    st.subheader("📊 Model Configurations")
    threshold = st.slider("Contradiction Filter Threshold", 0.0, 1.0, 0.5, 0.05,
                          help="Passages with a contradiction probability above this score are rejected.")

st.title("⚡ Enterprise Pre-Verification RAG Studio")
st.caption("Verify factual integrity and evaluate contexts before generating answers.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about your uploaded documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not st.session_state.vector_store:
        with st.chat_message("assistant"):
            st.warning("⚠️ Please upload and process documents in the sidebar first.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and verifying source context..."):
                vector_docs = st.session_state.vector_store.similarity_search(prompt, k=4)
                bm25_docs = st.session_state.bm25_retriever.invoke(prompt)
                
                combined_docs = {doc.page_content: doc for doc in (vector_docs + bm25_docs)}.values()
                
                verified_contexts = []
                verification_logs = []

                for i, doc in enumerate(combined_docs):
                    passage = doc.page_content
                    scores = verifier.predict([(prompt, passage)])[0]
                    
                    exp_scores = np.exp(scores)
                    probs = exp_scores / np.sum(exp_scores)
                    
                    contradiction_prob, entailment_prob, neutral_prob = probs[0], probs[1], probs[2]
                    
                    status = "Rejected" if contradiction_prob > threshold else "Retained"
                    
                    if status == "Retained":
                        verified_contexts.append(doc)
                    
                    verification_logs.append({
                        "index": i + 1,
                        "source": doc.metadata.get("source", "Unknown"),
                        "status": status,
                        "entailment": entailment_prob,
                        "contradiction": contradiction_prob,
                        "text": passage
                    })

            with st.expander("🛡️ Pre-Generation Context Verification Metrics", expanded=True):
                col1, col2 = st.columns(2)
                col1.metric("Retrieved Chunks", len(combined_docs))
                col2.metric("Verified Retained Chunks", len(verified_contexts))
                
                for log in verification_logs:
                    badge_class = "badge-retained" if log["status"] == "Retained" else "badge-rejected"
                    st.markdown(f"""
                    <div class="metric-card">
                        <span class="{badge_class}">{log['status']}</span> 
                        <strong>Source:</strong> {log['source']} | 
                        <strong>Entailment:</strong> {log['entailment']:.2f} | 
                        <strong>Contradiction:</strong> {log['contradiction']:.2f}
                        <p style="margin-top: 5px; font-size: 0.9rem;">{log['text'][:250]}...</p>
                    </div>
                    """, unsafe_allow_html=True)

            if verified_contexts:
                context_block = "\n\n".join([f"[{doc.metadata.get('source')}] {doc.page_content}" for doc in verified_contexts])
                
                response_text = f"### Answer Based on Verified Context\n\n"
                response_text += f"Found **{len(verified_contexts)} verified source passage(s)** matching your query:\n\n"
                response_text += f"> {context_block[:600]}...\n\n"
                response_text += "*Note: Low-confidence and contradictory passages were filtered out prior to answer compilation.*"
            else:
                response_text = "❌ **Verification Failed:** No passages passed the reliability and relevance threshold to safely answer your query."

            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
