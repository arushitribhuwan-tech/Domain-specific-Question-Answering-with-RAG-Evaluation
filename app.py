"""
RootIQ-Domain-QA-RAG: Main Streamlit Application.
Features:
- ChatGPT-style Sidebar with multi-session Chat History & '+ New Chat'
- Dynamic Document Upload (PDF, TXT, DOCX, MD) with instant indexing
- Strict single-document grounding (answers queries strictly from the active uploaded file)
- Verified Source Citations drawer (file name, page number, similarity score)
- RAGAS Evaluation Dashboard with benchmark scorecards and latency profiling
"""

import os
import time
import json
import uuid
import yaml
import pandas as pd
import streamlit as st

# Internal imports
from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import DocumentChunker
from src.embeddings.embed import EmbeddingService, VectorStoreService
from src.retrieval.retriever import Retriever
from src.generation.llm_service import LLMService
from src.evaluation.run_eval import EvaluationRunner

# Page Configuration
st.set_page_config(
    page_title="Domain QA Assistant - File-Specific RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

HISTORY_FILE = os.path.join("data", "chat_history.json")


def load_persisted_sessions() -> dict:
    """Loads chat sessions from local json storage."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_persisted_sessions(sessions: dict):
    """Saves chat sessions to local json storage."""
    try:
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(sessions, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.warning(f"Failed to persist chat history: {e}")


@st.cache_resource
def load_config():
    """Loads config.yaml safely."""
    config_path = "config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


@st.cache_resource
def init_services():
    """Initializes and caches core RAG pipeline services."""
    config = load_config()
    embed_cfg = config.get("embedding", {})
    paths_cfg = config.get("paths", {})
    retrieval_cfg = config.get("retrieval", {})
    llm_cfg = config.get("llm", {})

    embedding_service = EmbeddingService(
        model_name=embed_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        device=embed_cfg.get("device", "cpu")
    )

    vectorstore = VectorStoreService(
        persist_dir=paths_cfg.get("chroma_db_dir", "vectorstore/chroma_db"),
        embedding_service=embedding_service
    )

    # Ingest default academic regulations if vectorstore is completely empty
    if vectorstore.get_chunk_count() == 0:
        raw_dir = paths_cfg.get("raw_data_dir", "data/raw")
        loader = DocumentLoader(raw_data_dir=raw_dir)
        docs = loader.load_documents()
        if docs:
            chunker = DocumentChunker()
            chunks = chunker.chunk_documents(docs)
            vectorstore.index_chunks(chunks)

    retriever = Retriever(
        vectorstore_service=vectorstore,
        top_k=retrieval_cfg.get("top_k", 5),
        use_hybrid=retrieval_cfg.get("use_hybrid", False),
        use_reranker=retrieval_cfg.get("use_reranker", False)
    )

    llm_service = LLMService(
        ollama_url=llm_cfg.get("ollama_url", "http://localhost:11434"),
        model_name=llm_cfg.get("model_name", "llama3.2:3b"),
        temperature=llm_cfg.get("temperature", 0.1),
        timeout=llm_cfg.get("timeout_seconds", 30)
    )

    return vectorstore, retriever, llm_service


config = load_config()
vectorstore, retriever, llm_service = init_services()

# ----------------- SESSION STATE MANAGEMENT -----------------
if "sessions" not in st.session_state:
    loaded = load_persisted_sessions()
    if loaded:
        st.session_state.sessions = loaded
        st.session_state.active_session_id = list(loaded.keys())[0]
    else:
        initial_id = str(uuid.uuid4())[:8]
        st.session_state.sessions = {
            initial_id: {
                "title": "New Conversation",
                "created_at": time.strftime("%Y-%m-%d %H:%M"),
                "document_name": None,
                "messages": [
                    {
                        "role": "assistant",
                        "content": "👋 Hello! Upload a domain document (PDF, TXT, DOCX) in the sidebar to start asking questions strictly about it.",
                        "sources": [],
                        "latency_ms": 0
                    }
                ]
            }
        }
        st.session_state.active_session_id = initial_id

# Ensure active session exists
if st.session_state.active_session_id not in st.session_state.sessions:
    st.session_state.active_session_id = list(st.session_state.sessions.keys())[0]

current_session = st.session_state.sessions[st.session_state.active_session_id]

# ----------------- SIDEBAR (CHATGPT-STYLE) -----------------
with st.sidebar:
    st.markdown("### 🤖 Domain-Specific RAG")

    # "+ New Chat" Button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        new_id = str(uuid.uuid4())[:8]
        st.session_state.sessions[new_id] = {
            "title": "New Conversation",
            "created_at": time.strftime("%Y-%m-%d %H:%M"),
            "document_name": current_session.get("document_name"),
            "messages": [
                {
                    "role": "assistant",
                    "content": "👋 Started a new chat! Ask anything about the active document.",
                    "sources": [],
                    "latency_ms": 0
                }
            ]
        }
        st.session_state.active_session_id = new_id
        save_persisted_sessions(st.session_state.sessions)
        st.rerun()

    st.markdown("---")

    # Document Selection & Upload Section
    st.markdown("### 📄 Active Document")
    indexed_docs = vectorstore.get_all_source_documents()

    # File Uploader
    uploaded_file = st.file_uploader(
        "Upload a document (PDF, TXT, DOCX)",
        type=["pdf", "txt", "docx", "md"],
        help="The system will index this file and answer queries strictly from it."
    )

    if uploaded_file is not None:
        raw_dir = config.get("paths", {}).get("raw_data_dir", "data/raw")
        os.makedirs(raw_dir, exist_ok=True)
        save_path = os.path.join(raw_dir, uploaded_file.name)

        if not os.path.exists(save_path):
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner(f"Indexing '{uploaded_file.name}'..."):
                loader = DocumentLoader(raw_data_dir=raw_dir)
                doc_sections = loader.load_single_file(save_path, uploaded_file.name)
                if doc_sections:
                    chunker = DocumentChunker()
                    new_chunks = chunker.chunk_documents(doc_sections)
                    vectorstore.index_chunks(new_chunks)
                    st.success(f"Indexed {len(new_chunks)} chunks from '{uploaded_file.name}'!")
                    current_session["document_name"] = uploaded_file.name
                    save_persisted_sessions(st.session_state.sessions)
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(f"Could not extract text from '{uploaded_file.name}'.")

    # Dropdown to choose among available indexed documents
    indexed_docs = vectorstore.get_all_source_documents()
    active_doc = current_session.get("document_name")

    if indexed_docs:
        # Default to current session doc or first indexed doc
        default_idx = indexed_docs.index(active_doc) if active_doc in indexed_docs else 0
        selected_doc = st.selectbox(
            "Query Specific Document:",
            options=indexed_docs,
            index=default_idx
        )
        if selected_doc != active_doc:
            current_session["document_name"] = selected_doc
            save_persisted_sessions(st.session_state.sessions)
            st.rerun()
        active_doc = selected_doc
        st.info(f"🎯 **Target:** `{active_doc}`")
    else:
        st.warning("⚠️ No documents indexed yet. Upload a file above!")

    st.markdown("---")

    # ChatGPT-style Chat History Section
    st.markdown("### 💬 Chat History")
    session_keys = list(st.session_state.sessions.keys())

    # Display session list with click to activate
    for s_id in reversed(session_keys):
        sess = st.session_state.sessions[s_id]
        title = sess.get("title", "Conversation")
        is_active = (s_id == st.session_state.active_session_id)

        col_btn, col_del = st.columns([5, 1])
        with col_btn:
            label = f"👉 {title}" if is_active else f"💬 {title}"
            if st.button(label, key=f"sess_btn_{s_id}", use_container_width=True):
                st.session_state.active_session_id = s_id
                st.rerun()

        with col_del:
            if len(session_keys) > 1:
                if st.button("🗑️", key=f"del_btn_{s_id}", help="Delete chat"):
                    del st.session_state.sessions[s_id]
                    if st.session_state.active_session_id == s_id:
                        st.session_state.active_session_id = list(st.session_state.sessions.keys())[0]
                    save_persisted_sessions(st.session_state.sessions)
                    st.rerun()

    if st.button("🧹 Clear All Chats", use_container_width=True):
        st.session_state.sessions = {}
        new_id = str(uuid.uuid4())[:8]
        st.session_state.sessions[new_id] = {
            "title": "New Conversation",
            "created_at": time.strftime("%Y-%m-%d %H:%M"),
            "document_name": active_doc,
            "messages": [
                {
                    "role": "assistant",
                    "content": "👋 Fresh chat started! Ask any question about your document.",
                    "sources": [],
                    "latency_ms": 0
                }
            ]
        }
        st.session_state.active_session_id = new_id
        save_persisted_sessions(st.session_state.sessions)
        st.rerun()

    st.markdown("---")

    # Pipeline Settings & System Health
    st.markdown("### ⚙️ Engine Settings")
    groq_api_key = st.text_input(
        "🔑 Groq API Key (Recommended - Free)",
        type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        help="Paste a free Groq API key from https://console.groq.com/keys for instant, highly accurate Llama-3.3-70B answers!"
    )
    if not (groq_api_key and len(groq_api_key.strip()) > 10):
        st.markdown("👉 *[Click here to get a free Groq Key](https://console.groq.com/keys) for full AI answers!*")

    top_k = st.slider("Top-K Retrieval", min_value=1, max_value=10, value=config.get("retrieval", {}).get("top_k", 5))
    temperature = st.slider("LLM Temperature", min_value=0.0, max_value=1.0, value=float(config.get("llm", {}).get("temperature", 0.1)), step=0.05)

    ollama_online = llm_service.check_ollama_status()
    if groq_api_key and len(groq_api_key.strip()) > 10:
        st.success("🟢 Engine: Groq Cloud (Llama-3.3-70B)")
    elif ollama_online:
        st.success(f"🟢 Engine: Local Ollama (`{llm_service.model_name}`)")
    else:
        st.warning("🟡 Engine: Grounded Synthesizer (Ollama offline)")
    st.caption(f"💾 Total Indexed Chunks: {vectorstore.get_chunk_count()}")


# ----------------- MAIN CONTENT AREA -----------------
tab_chat, tab_eval = st.tabs(["💬 Domain Q&A Chatbot", "📊 RAGAS Evaluation Dashboard"])

# ==================== TAB 1: CHATBOT ====================
with tab_chat:
    active_doc = current_session.get("document_name")

    if not active_doc:
        st.title("🎓 Domain-Specific Question Answering")
        st.info("👈 **To get started:** Please upload a document (PDF, TXT, DOCX) in the sidebar, or select one from your files.")

        col_sample1, col_sample2 = st.columns([1, 2])
        with col_sample1:
            if st.button("📚 Load Sample Academic Handbook", type="primary"):
                sample_file = "academic_regulations_handbook.txt"
                raw_dir = config.get("paths", {}).get("raw_data_dir", "data/raw")
                sample_path = os.path.join(raw_dir, sample_file)
                if os.path.exists(sample_path):
                    loader = DocumentLoader(raw_data_dir=raw_dir)
                    docs = loader.load_single_file(sample_path, sample_file)
                    chunker = DocumentChunker()
                    chunks = chunker.chunk_documents(docs)
                    vectorstore.index_chunks(chunks)
                    current_session["document_name"] = sample_file
                    save_persisted_sessions(st.session_state.sessions)
                    st.rerun()
    else:
        st.title("🎓 Domain-Specific Question Answering")
        st.markdown(f"**Target Document:** 📄 `{active_doc}` • *Answers are grounded strictly in this document.*")

        # Display conversation messages for the current session
        for msg in current_session["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander(f"📚 Verified Sources ({len(msg['sources'])}) | Latency: {msg.get('latency_ms', 0)}ms"):
                        for idx, src in enumerate(msg["sources"]):
                            st.markdown(
                                f"**Source {idx+1}:** `{src.get('source_document')}` (Page {src.get('page_number', 1)}) | "
                                f"Relevance Score: `{src.get('score', 0.0):.2f}`"
                            )
                            st.caption(f"\"{src.get('snippet', '')}\"")

        # Chat Input Box
        user_query = st.chat_input(f"Ask any question about {active_doc}...")

        if user_query:
            # Append user question
            current_session["messages"].append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            # Auto-title conversation on first question
            if current_session.get("title") == "New Conversation":
                clean_title = user_query.strip().capitalize()
                current_session["title"] = clean_title[:32] + "..." if len(clean_title) > 32 else clean_title

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner(f"Searching `{active_doc}` and generating grounded answer..."):
                    t0 = time.time()
                    retrieved_chunks = retriever.retrieve(user_query, k=top_k, filter_document=active_doc)
                    retrieval_ms = int((time.time() - t0) * 1000)

                    t1 = time.time()
                    llm_service.temperature = temperature
                    response = llm_service.generate_answer(user_query, retrieved_chunks, groq_key_override=groq_api_key)
                    generation_ms = int((time.time() - t1) * 1000)

                    total_ms = retrieval_ms + generation_ms
                    answer_text = response["answer"]
                    sources = response.get("sources", [])

                    st.markdown(answer_text)
                    st.caption(f"⚡ Latency: {total_ms}ms | Engine: `{response.get('model_used', 'RAG')}`")

                    if sources:
                        with st.expander(f"📚 Verified Sources ({len(sources)}) | Retrieval: {retrieval_ms}ms | Generation: {generation_ms}ms"):
                            for idx, src in enumerate(sources):
                                st.markdown(
                                    f"**Source {idx+1}:** `{src.get('source_document')}` (Page {src.get('page_number', 1)}) | "
                                    f"Score: `{src.get('score', 0.0):.2f}`"
                                )
                                st.caption(f"\"{src.get('snippet', '')}\"")

            # Save assistant message
            current_session["messages"].append({
                "role": "assistant",
                "content": answer_text,
                "sources": sources,
                "latency_ms": total_ms
            })

            # Persist to disk
            save_persisted_sessions(st.session_state.sessions)


# ==================== TAB 2: EVALUATION DASHBOARD ====================
with tab_eval:
    st.title("📊 RAG Pipeline Quantitative Evaluation")
    st.markdown("Automatic benchmarking using **RAGAS** quality metrics and latency measurements across the curated test set.")

    eval_results_file = config.get("paths", {}).get("eval_results_file", "data/eval/eval_results.csv")
    test_qa_file = config.get("paths", {}).get("test_qa_file", "data/eval/test_qa.csv")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.info(f"Target QA Benchmark File: `{test_qa_file}`")
    with col2:
        run_button = st.button("🚀 Run Batch Evaluation", use_container_width=True)

    if run_button:
        with st.spinner("Running batch evaluation on test set... Please wait."):
            runner = EvaluationRunner()
            df_eval = runner.run()
            st.success(f"Evaluation finished! Evaluated {len(df_eval)} questions.")

    if os.path.exists(eval_results_file):
        df = pd.read_csv(eval_results_file)

        # Top metric cards
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Context Precision", f"{df['context_precision'].mean():.2f}")
        m2.metric("Context Recall", f"{df['context_recall'].mean():.2f}")
        m3.metric("Faithfulness", f"{df['faithfulness'].mean():.2f}")
        m4.metric("Answer Relevancy", f"{df['answer_relevancy'].mean():.2f}")
        m5.metric("Correctness", f"{df['answer_correctness'].mean():.2f}")
        m6.metric("Avg Latency", f"{df['total_latency_ms'].mean():.0f} ms")

        st.subheader("Metric Distributions")
        metrics_cols = ["context_precision", "context_recall", "faithfulness", "answer_relevancy", "answer_correctness"]
        st.bar_chart(df[metrics_cols].mean())

        st.subheader("Detailed Evaluation Results")
        st.dataframe(df, use_container_width=True)

        # Download CSV button
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Evaluation Report (CSV)",
            data=csv_data,
            file_name="rag_evaluation_report.csv",
            mime="text/csv"
        )
    else:
        st.warning("No evaluation results found yet. Click 'Run Batch Evaluation' above to run benchmarks.")
