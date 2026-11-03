import json
import os
import sys
import tempfile
import urllib.request
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, "python"))
sys.path.append(os.path.join(ROOT, "evaluation"))
from models.document import Document  # noqa: E402
from cas_engine import CASEngine  # noqa: E402
from services.nli_model_verifier import NLIModelVerifier  # noqa: E402
# Same prompt, decoding settings and passage format as the experiments
from run_llm_generation import OPTIONS, QA_PROMPT, format_passages  # noqa: E402

THETA = 0.99   # contradiction threshold chosen by cross-validation and used in every experiment
EXPLAINED_PROMPT = """Answer the question using only the passages below, citing the passages you use, e.g. [2].
If the passages contradict each other, say so and give each position with its citation.
If the passages do not contain the answer, reply exactly: unknown

{passages}

Question: {question}
Answer:"""
OLLAMA_URL = "http://localhost:11434"


def ollama_models():
    """Names of the models installed in the local Ollama server ([] if it is not running)."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as resp:
            return [m["name"] for m in json.load(resp)["models"]]
    except OSError:
        return []


def ollama_stream(model, prompt, options):
    """Stream an answer from the local Ollama server, piece by piece."""
    body = json.dumps({"model": model, "prompt": prompt, "stream": True, "options": options}).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/generate", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for line in resp:
            if line.strip():
                chunk = json.loads(line)
                if "error" in chunk:
                    raise RuntimeError(chunk["error"])
                yield chunk.get("response", "")

st.set_page_config(
    page_title="CAS-V RAG Verification Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Complete theme overrides
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Force global light mode backgrounds */
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

    /* Sidebar and File Uploader */
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
    }

    [data-testid="stFileUploader"] *,
    [data-testid="stFileUploadDropzone"] * {
        color: #334155 !important;
    }

    /* Action Buttons */
    .stButton > button {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    /* Sticky Bottom Input Bar */
    div[data-testid="stBottom"],
    div[data-testid="stBottomBlockContainer"] {
        background-color: #ffffff !important;
        border-top: 1px solid #e2e8f0 !important;
    }

    /* Input Field Background and Dark Typing Text */
    [data-testid="stChatInput"],
    [data-testid="stChatInputContainer"],
    div[data-baseweb="base-input"],
    div[data-baseweb="input"] {
        background-color: #f8fafc !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 12px !important;
    }

    textarea[data-testid="stChatInputTextArea"],
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInputContainer"] textarea,
    div[data-baseweb="base-input"] textarea {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        caret-color: #2563eb !important;
        background-color: transparent !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        opacity: 1 !important;
    }

    /* Visible Arrow Submit Button */
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

    /* Metric Cards */
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
    threshold = st.slider("Trust Threshold (τ)", 0.10, 0.80, 0.45, 0.05,
                          help="Chunks whose CAS trust score falls below τ are rejected.")
    conflict_mode = st.radio(
        "Conflict handling", ["Conflict-preserving", "Contradiction penalty"],
        help="Conflict-preserving keeps both sides of an on-topic contradiction and labels it; "
             "Contradiction penalty down-weights chunks that contradict others.")

    st.markdown("---")
    st.subheader("🤖 Answer Generation (Ollama)")
    installed = ollama_models()
    if installed:
        llm_model = st.selectbox("LLM", installed,
                                 index=installed.index("llama3.2:latest") if "llama3.2:latest" in installed else 0)
    else:
        llm_model = None
        st.error("Ollama is not running. Start it with `ollama serve`, then reload this page.")
    answer_style = st.radio(
        "Answer style", ["Short (as in the experiments)", "Explained, with citations"],
        help="Short uses the exact prompt and settings of the evaluation; Explained allows a longer, cited answer.")

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
                
                combined_docs = list({doc.page_content: doc for doc in (vector_docs + bm25_docs)}.values())

                # CAS pipeline: pairwise passage NLI -> conflict graph -> trust scoring -> synthesis.
                # Retrieval order stands in for relevance; uploaded files carry no provenance metadata (prior 0.5).
                n = len(combined_docs)
                cas_docs = [
                    Document(id=f"C{i + 1}", content=doc.page_content,
                             source=doc.metadata.get("source", "Unknown"),
                             retrieval_confidence=1 - i / n, relevance_score=1 - i / n, provenance_score=0.5)
                    for i, doc in enumerate(combined_docs)
                ]
                engine = CASEngine(trust_threshold=threshold,
                                   verifier=NLIModelVerifier(model=verifier, contradiction_threshold=THETA),
                                   preserve_conflicts=conflict_mode == "Conflict-preserving")
                result = engine.execute_pipeline(prompt, cas_docs)
                graph = result["graph"]
                verified_contexts = [v for v in result["verified_documents"] if not v.is_filtered]

            with st.expander("🛡️ Pre-Generation Context Verification Metrics", expanded=True):
                col1, col2, col3 = st.columns(3)
                col1.metric("Retrieved Chunks", len(combined_docs))
                col2.metric("Verified Retained Chunks", len(verified_contexts))
                col3.metric("Contradicting Pairs", sum(e.relation.value == "Contradiction" for e in graph.edges))

                for v in result["verified_documents"]:
                    status = "Rejected" if v.is_filtered else "Retained"
                    badge_class = "badge-retained" if status == "Retained" else "badge-rejected"
                    st.markdown(f"""
                    <div class="metric-card">
                        <span class="{badge_class}">{status}</span>
                        <strong style="color: #0f172a;">{v.doc.id}</strong> |
                        <strong style="color: #0f172a;">Source:</strong> {v.doc.source} |
                        <strong style="color: #0f172a;">Trust:</strong> {v.trust_score:.2f} |
                        <strong style="color: #0f172a;">Agreement:</strong> {graph.get_agreement_score(v.doc.id):.2f} |
                        <strong style="color: #0f172a;">Contradiction:</strong> {v.conflict_penalty:.2f}
                        <p style="margin-top: 8px; font-size: 0.9rem; color: #475569;">{v.doc.content[:250]}...</p>
                    </div>
                    """, unsafe_allow_html=True)

            if not verified_contexts:
                response_text = "❌ **Verification Failed:** No passages passed the reliability and relevance threshold to safely answer your query."
                st.markdown(response_text)
            elif llm_model is None:
                response_text = "⚠️ **No answer generated:** Ollama is not running. Start it with `ollama serve`."
                st.markdown(response_text)
            else:
                # The LLM sees the kept chunks in retrieval order plus a note per kept contradiction, as in the experiments
                index = {d.id: i for i, d in enumerate(cas_docs)}
                kept = sorted(index[v.doc.id] for v in verified_contexts)
                conflicts = [(index[a], index[b]) for a, b in result["conflicts"]]
                context = format_passages([d.content for d in cas_docs], kept, conflicts)
                short = answer_style.startswith("Short")
                llm_prompt = (QA_PROMPT if short else EXPLAINED_PROMPT).format(passages=context, question=prompt)
                options = OPTIONS if short else {**OPTIONS, "num_predict": 512}
                try:
                    answer = st.write_stream(ollama_stream(llm_model, llm_prompt, options)).strip()
                except Exception as e:
                    answer = f"⚠️ **{llm_model} failed:** {e}"
                    st.markdown(answer)
                legend = " · ".join(f"[{k + 1}] {cas_docs[i].id} ({cas_docs[i].source})" for k, i in enumerate(kept))
                notes = ""
                if result["conflicts"]:
                    pairs = ", ".join(f"{a} ↔ {b}" for a, b in result["conflicts"])
                    notes = f"⚠️ **Contradicting passages passed to the model:** {pairs}\n\n"
                footer = f"{notes}*{llm_model} answered from {len(kept)} verified passage(s): {legend}*"
                st.markdown(footer)
                response_text = f"{answer}\n\n{footer}"
            st.session_state.messages.append({"role": "assistant", "content": response_text})
