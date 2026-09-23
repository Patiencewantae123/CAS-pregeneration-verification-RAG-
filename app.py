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
<<<<<<< Updated upstream
    page_title="Enterprise RAG Verification Studio",
=======
    page_title="CAS-V RAG Verification Studio",
>>>>>>> Stashed changes
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Forceful CSS fix for text visibility, cursor, and placeholder contrast
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Universal Light Background */
    .stApp, 
    [data-testid="stSidebar"], 
    [data-testid="stHeader"], 
    [data-testid="stMainBlockContainer"],
    [data-testid="stBottom"],
    [data-testid="stBottomBlockContainer"],
    footer {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }

    /* File Uploader Container & Inner Elements */
    [data-testid="stFileUploader"],
    [data-testid="stFileUploadDropzone"],
    section[data-testid="stFileUploaderDropzone"] {
        background-color: #f8fafc !important;
        border: 1px dashed #cbd5e1 !important;
        border-radius: 12px !important;
    }

    [data-testid="stFileUploader"] button,
    [data-testid="stFileUploadDropzone"] button,
    section[data-testid="stFileUploaderDropzone"] button {
        background-color: #e2e8f0 !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        box-shadow: none !important;
    }

    [data-testid="stFileUploader"] *,
    [data-testid="stFileUploadDropzone"] * {
        color: #334155 !important;
    }

    /* Primary Sidebar Buttons */
    .stButton > button {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1rem !important;
    }

    /* Bottom Chat Container */
    div[data-testid="stBottom"],
    div[data-testid="stBottomBlockContainer"] {
        background-color: #ffffff !important;
        border-top: 1px solid #e2e8f0 !important;
    }

    /* Chat Input Box Wrapper */
    [data-testid="stChatInput"],
    [data-testid="stChatInputContainer"],
    div[data-baseweb="base-input"],
    div[data-baseweb="input"] {
        background-color: #f8fafc !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 12px !important;
    }

    /* CRITICAL FIX: Explicitly override webkit fill color, text color, and typing cursor */
    textarea[data-testid="stChatInputTextArea"],
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInputContainer"] textarea,
    div[data-baseweb="base-input"] textarea,
    div[data-baseweb="input"] textarea {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        caret-color: #2563eb !important;
        background-color: transparent !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        opacity: 1 !important;
    }

    /* Visible Placeholder Text */
    [data-testid="stChatInput"] textarea::placeholder,
    textarea[data-testid="stChatInputTextArea"]::placeholder {
        color: #64748b !important;
        -webkit-text-fill-color: #64748b !important;
        opacity: 1 !important;
    }

    /* Visible Send Arrow Button */
    [data-testid="stChatInput"] button,
    [data-testid="stChatInputSubmitButton"] {
        background-color: #2563eb !important;
        border-radius: 8px !important;
        border: none !important;
        opacity: 1 !important;
    }
    
    [data-testid="stChatInput"] button svg,
    [data-testid="stChatInputSubmitButton"] svg {
        fill: #ffffff !important;
        color: #ffffff !important;
        stroke: #ffffff !important;
    }

    /* Metrics & Cards */
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }

    .badge-retained {
        background-color: #dcfce7;
        color: #15803d;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.78rem;
    }
    .badge-rejected {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.78rem;
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

st.title("⚡ CAS-V Pre-Verification RAG Studio")
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
                        <strong style="color: #0f172a;">Source:</strong> {log['source']} | 
                        <strong style="color: #0f172a;">Entailment:</strong> {log['entailment']:.2f} | 
                        <strong style="color: #0f172a;">Contradiction:</strong> {log['contradiction']:.2f}
                        <p style="margin-top: 8px; font-size: 0.9rem; color: #475569;">{log['text'][:250]}...</p>
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