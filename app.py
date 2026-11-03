import codecs
import html
import io
import json
import os
import sys
import time
import urllib.request
import streamlit as st

from charset_normalizer import from_bytes
from langchain_core.documents import Document as LCDocument
from pypdf import PdfReader
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
from run_evaluation import bm25_scores  # noqa: E402

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
THETA = 0.99   # contradiction threshold chosen by cross-validation and used in every experiment
EXPLAINED_PROMPT = """Answer the question using only the passages below, citing the passages you use, e.g. [2].
If the passages contradict each other, say so and give each position with its citation.
If the passages do not contain the answer, reply exactly: unknown

{passages}

Question: {question}
Answer:"""
OLLAMA_URL = "http://localhost:11434"
NLI_CACHE = os.path.join(ROOT, ".cas_cache", "app_nli_cache.jsonl")   # kept apart from the experiments' cache


def decode_text(data):
    """Text of a TXT upload: a byte-order mark wins, then UTF-8, then the detected encoding (e.g. CP949, CP1252)."""
    for bom, encoding in ((codecs.BOM_UTF8, "utf-8-sig"), (codecs.BOM_UTF16_LE, "utf-16"), (codecs.BOM_UTF16_BE, "utf-16")):
        if data.startswith(bom):
            text = data.decode(encoding)
            break
    else:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            best = from_bytes(data).best()
            if best is None:
                raise ValueError("unknown text encoding; save the file as UTF-8")
            text = str(best)
    if "\x00" in text:
        raise ValueError("this looks like a binary file, not plain text")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def load_file(uploaded_file):
    """Pages of an uploaded PDF (read in memory), or the text of a TXT file."""
    data = uploaded_file.getvalue()
    if uploaded_file.name.lower().endswith(".pdf"):
        docs = [LCDocument(page_content=page.extract_text() or "", metadata={"page": i})
                for i, page in enumerate(PdfReader(io.BytesIO(data)).pages)]
    else:
        docs = [LCDocument(page_content=decode_text(data))]
    for d in docs:
        d.metadata["source"] = uploaded_file.name
    return docs


def chunk_card(c, text_chars=250):
    """HTML card for one chunk; document text is escaped so it cannot break the page."""
    badge = (f'<span class="badge-{"rejected" if c["status"] == "Rejected" else "retained"}">{c["status"]}</span> '
             if c.get("status") else "")
    scores = (f' | <strong>Trust:</strong> {c["trust"]:.2f} | <strong>Agreement:</strong> {c["agreement"]:.2f}'
              f' | <strong>Contradiction:</strong> {c["contradiction"]:.2f}' if "trust" in c else "")
    # Escaped, and line breaks as <br>: a blank line would otherwise end the HTML block and render the rest as Markdown
    text = html.escape(c["text"][:text_chars]).replace("\n", "<br>") + ("..." if len(c["text"]) > text_chars else "")
    return (f'<div class="metric-card">{badge}<strong>{c["id"]}</strong> | <strong>Source:</strong> '
            f'{html.escape(c["source"])}{scores}'
            f'<p style="margin-top: 8px; font-size: 0.9rem; color: #475569;">{text}</p></div>')


def show_details(d, expanded):
    """Verification panel for one answer: contradictions side by side, every chunk's decision, timings."""
    with st.expander("🛡️ Pre-Generation Context Verification", expanded=expanded):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Retrieved Chunks", d["retrieved"])
        col2.metric("Kept after Verification", d["kept"])
        col3.metric("Contradicting Pairs", len(d["conflicts"]))
        col4.metric("Conflict Check Time", f'{d["timings"]["Conflict check"]:.1f} s')
        st.caption(" · ".join(f"{k}: {v:.1f} s" for k, v in d["timings"].items()))
        for c in d["conflicts"]:
            if not c["both_kept"]:
                status = "not both kept"
            elif d["preserve"]:
                status = "both kept, with a conflict note for the LLM"
            else:
                status = "both kept, without a note (penalty mode)"
            st.markdown(f"**⚔️ {c['a']['id']} contradicts {c['b']['id']}** (confidence {c['confidence']:.2f}) · {status}")
            left, right = st.columns(2)
            left.markdown(chunk_card(c["a"], 600), unsafe_allow_html=True)
            right.markdown(chunk_card(c["b"], 600), unsafe_allow_html=True)
        for c in d["cards"]:
            st.markdown(chunk_card(c), unsafe_allow_html=True)


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
def load_verifier(model_name, theta, cache_path):
    """NLI cross-encoder at threshold θ. Predictions are cached on disk, keyed by model name, so chunk pairs seen
    before cost nothing. The arguments are part of Streamlit's cache key, so changing any of them reloads."""
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    except OSError:   # read-only install: keep predictions in memory only
        cache_path = None
    return NLIModelVerifier(model_name=model_name, model=CrossEncoder(model_name), contradiction_threshold=theta,
                            cache_path=cache_path)

@st.cache_resource(show_spinner="Loading Embedding Model...")
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

verifier = load_verifier(NLI_MODEL, THETA, NLI_CACHE)
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
                    try:
                        docs = load_file(uploaded_file)
                    except Exception as e:
                        st.warning(f"Skipped {uploaded_file.name}: {e}")
                        continue
                    if not any(d.page_content.strip() for d in docs):
                        st.warning(f"Skipped {uploaded_file.name}: no extractable text (a scanned PDF needs OCR first).")
                        continue
                    all_docs.extend(docs)

                splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                chunks = [c for c in splitter.split_documents(all_docs) if c.page_content.strip()]
                if chunks:
                    st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
                    st.session_state.bm25_retriever = BM25Retriever.from_documents(chunks)
                    st.session_state.bm25_retriever.k = 5
                    st.session_state.all_chunks = chunks
                    n_files = len({c.metadata["source"] for c in chunks})
                    st.success(f"Indexed {len(chunks)} chunks across {n_files} document(s)!")
                else:
                    # Drop any earlier index too, so questions are not silently answered from old documents
                    st.session_state.vector_store = st.session_state.bm25_retriever = None
                    st.session_state.all_chunks = []
                    st.error("No text could be indexed from these files. The previous index was cleared.")

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

    st.markdown("---")
    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("⚡ CAS-V Pre-Verification RAG Studio")
st.caption("Verify factual integrity and evaluate contexts before generating answers.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("details"):
            show_details(msg["details"], expanded=False)
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
            timings = {}
            with st.spinner("Retrieving and verifying source context..."):
                start = time.perf_counter()
                vector_docs = st.session_state.vector_store.similarity_search(prompt, k=4)
                bm25_docs = st.session_state.bm25_retriever.invoke(prompt)
                combined_docs = list({doc.page_content: doc for doc in (vector_docs + bm25_docs)}.values())
                timings["Retrieval"] = time.perf_counter() - start

                start = time.perf_counter()
                # CAS pipeline: pairwise passage NLI -> conflict graph -> trust scoring -> synthesis.
                # Scored as in the experiments: relevance = BM25 / pool maximum, confidence from the BM25 rank.
                # Uploaded files carry no provenance metadata (prior 0.5).
                texts = [doc.page_content for doc in combined_docs]
                raw = bm25_scores(prompt, texts)
                hi = max(raw)
                rank = {i: r for r, i in enumerate(sorted(range(len(texts)), key=lambda i: -raw[i]))}
                n = len(texts)
                cas_docs = [
                    Document(id=f"C{i + 1}", content=text, source=doc.metadata.get("source", "Unknown"),
                             retrieval_confidence=1 - rank[i] / n, relevance_score=raw[i] / hi if hi > 0 else 0.5,
                             provenance_score=0.5)
                    for i, (text, doc) in enumerate(zip(texts, combined_docs))
                ]
                engine = CASEngine(trust_threshold=threshold, verifier=verifier,
                                   preserve_conflicts=conflict_mode == "Conflict-preserving")
                result = engine.execute_pipeline(prompt, cas_docs)
                graph = result["graph"]
                verified_contexts = [v for v in result["verified_documents"] if not v.is_filtered]
                timings["Conflict check"] = time.perf_counter() - start

            def side(doc):
                return {"id": doc.id, "source": doc.source, "text": doc.content}
            kept_ids = {v.doc.id for v in verified_contexts}
            details = {
                "retrieved": len(combined_docs), "kept": len(verified_contexts), "timings": timings,
                "preserve": conflict_mode == "Conflict-preserving",
                "cards": [{**side(v.doc), "status": "Rejected" if v.is_filtered else "Retained", "trust": v.trust_score,
                           "agreement": graph.get_agreement_score(v.doc.id), "contradiction": v.conflict_penalty}
                          for v in result["verified_documents"]],
                "conflicts": [{"a": side(graph.nodes[e.source_doc_id]), "b": side(graph.nodes[e.target_doc_id]),
                               "confidence": e.confidence,
                               "both_kept": {e.source_doc_id, e.target_doc_id} <= kept_ids}
                              for e in graph.edges if e.relation.value == "Contradiction"],
            }
            show_details(details, expanded=True)

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
                start = time.perf_counter()
                try:
                    answer = st.write_stream(ollama_stream(llm_model, llm_prompt, options)).strip()
                    failed = False
                except Exception as e:
                    answer, failed = f"⚠️ **{llm_model} failed:** {e}", True
                    st.markdown(answer)
                timings["Answer"] = time.perf_counter() - start
                if failed:
                    response_text = answer
                else:
                    legend = " · ".join(f"[{k + 1}] {cas_docs[i].id} ({cas_docs[i].source})" for k, i in enumerate(kept))
                    notes = ""
                    if result["conflicts"]:
                        pairs = ", ".join(f"{a} ↔ {b}" for a, b in result["conflicts"])
                        notes = f"⚠️ **Contradicting passages passed to the model:** {pairs}\n\n"
                    footer = (f"{notes}*{llm_model} answered in {timings['Answer']:.1f} s from {len(kept)} "
                              f"verified passage(s): {legend}*")
                    st.markdown(footer)
                    response_text = f"{answer}\n\n{footer}"
            st.session_state.messages.append({"role": "assistant", "content": response_text, "details": details})
