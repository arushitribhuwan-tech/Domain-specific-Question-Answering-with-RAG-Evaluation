"""
RootIQ-Domain-QA-RAG: Main Streamlit Application.
Features:
- Interactive grounded Chat Interface with streaming/spinner state
- Collapsible Source Citations panel (document name, page number, similarity score)
- Evaluation Dashboard with RAGAS metric scorecards and latency charts
- Document Management & Ingestion Panel (upload files, trigger re-indexing)
"""

import os
import time
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
    page_title="Domain QA with RAG Evaluation",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


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

    # Auto-ingest if vectorstore is empty
    if vectorstore.get_chunk_count() == 0:
        raw_dir = paths_cfg.get("raw_data_dir", "data/raw")
        loader = DocumentLoader(raw_data_dir=raw_dir)
        docs = loader.load_documents()
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

# ----------------- SIDEBAR -----------------
st.sidebar.title("⚙️ System Configuration")
domain_name = config.get("project", {}).get("domain", "Academic Regulations")
st.sidebar.markdown(f"**Domain:** `{domain_name}`")

# Model & Parameter Controls
st.sidebar.subheader("Pipeline Settings")
top_k = st.sidebar.slider("Top-K Retrieval", min_value=1, max_value=10, value=config.get("retrieval", {}).get("top_k", 5))
temperature = st.sidebar.slider("LLM Temperature", min_value=0.0, max_value=1.0, value=float(config.get("llm", {}).get("temperature", 0.1)), step=0.05)

# System Health & Status
st.sidebar.subheader("System Status")
ollama_online = llm_service.check_ollama_status()
if ollama_online:
    st.sidebar.success("🟢 Ollama LLM Runtime: Online")
else:
    st.sidebar.warning("🟡 Ollama: Offline (Extractive Fallback Active)")

chunk_count = vectorstore.get_chunk_count()
st.sidebar.info(f"Indexed Chunks: **{chunk_count}**")

# ----------------- MAIN NAVIGATION -----------------
tabs = st.tabs(["💬 Domain Q&A Chatbot", "📊 RAGAS Evaluation Dashboard", "📁 Document Management & Ingestion"])

# ==================== TAB 1: CHATBOT ====================
with tabs[0]:
    st.title("🎓 Domain-Specific Question Answering Assistant")
    st.caption("Reliable, source-grounded answers powered by RAG with strict evidence citations.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I am your domain assistant. Ask me questions about academic policies, admissions, exams, or scholarships.",
                "sources": [],
                "latency_ms": 0
            }
        ]

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📚 Verified Sources ({len(msg['sources'])}) | Latency: {msg.get('latency_ms', 0)}ms"):
                    for idx, src in enumerate(msg["sources"]):
                        st.markdown(
                            f"**Source {idx+1}:** `{src.get('source_document')}` (Page {src.get('page_number', 1)}) | "
                            f"Similarity: `{src.get('score', 0.0):.2f}`"
                        )
                        st.caption(f"\"{src.get('snippet', '')}\"")

    # Chat Input Box
    user_query = st.chat_input("Ask a question about the regulations...")

    if user_query:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate response with spinner
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant documents and generating answer..."):
                t0 = time.time()
                retrieved_chunks = retriever.retrieve(user_query, k=top_k)
                retrieval_ms = int((time.time() - t0) * 1000)

                t1 = time.time()
                response = llm_service.generate_answer(user_query, retrieved_chunks)
                generation_ms = int((time.time() - t1) * 1000)

                total_ms = retrieval_ms + generation_ms
                answer_text = response["answer"]
                sources = response.get("sources", [])

                st.markdown(answer_text)

                if sources:
                    with st.expander(f"📚 Verified Sources ({len(sources)}) | Retrieval: {retrieval_ms}ms | Generation: {generation_ms}ms"):
                        for idx, src in enumerate(sources):
                            st.markdown(
                                f"**Source {idx+1}:** `{src.get('source_document')}` (Page {src.get('page_number', 1)}) | "
                                f"Score: `{src.get('score', 0.0):.2f}`"
                            )
                            st.caption(f"\"{src.get('snippet', '')}\"")

        # Save to session history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "sources": sources,
            "latency_ms": total_ms
        })


# ==================== TAB 2: EVALUATION DASHBOARD ====================
with tabs[1]:
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


# ==================== TAB 3: DOCUMENT MANAGEMENT ====================
with tabs[2]:
    st.title("📁 Document Ingestion & Vector Indexing")
    st.markdown("Manage domain source documents in `data/raw/` and rebuild the ChromaDB vector index.")

    raw_dir = config.get("paths", {}).get("raw_data_dir", "data/raw")
    os.makedirs(raw_dir, exist_ok=True)

    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("Existing Domain Documents")
        files = os.listdir(raw_dir)
        if files:
            for f in files:
                st.write(f"📄 **{f}** ({os.path.getsize(os.path.join(raw_dir, f))} bytes)")
        else:
            st.write("No files in `data/raw/` yet.")

    with c2:
        st.subheader("Upload New Document")
        uploaded_file = st.file_uploader("Upload PDF or TXT notice/handbook", type=["pdf", "txt", "md"])
        if uploaded_file is not None:
            save_path = os.path.join(raw_dir, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Uploaded `{uploaded_file.name}` to `data/raw/`.")

    st.markdown("---")
    st.subheader("Vector Database Indexing")
    st.caption("Re-reading all files in `data/raw/`, chunking with overlap, generating embeddings, and storing in ChromaDB.")

    if st.button("🔄 Trigger Re-Indexing Pipeline", type="primary"):
        with st.spinner("Extracting, cleaning, chunking, and embedding documents..."):
            loader = DocumentLoader(raw_data_dir=raw_dir)
            docs = loader.load_documents()

            ingestion_cfg = config.get("ingestion", {})
            chunker = DocumentChunker(
                chunk_size=ingestion_cfg.get("chunk_size", 400),
                chunk_overlap=ingestion_cfg.get("chunk_overlap", 50)
            )
            chunks = chunker.chunk_documents(docs)

            vectorstore.reset_index()
            indexed_count = vectorstore.index_chunks(chunks)

            st.success(f"✅ Ingestion complete! Indexed **{indexed_count}** chunks into ChromaDB.")
            st.rerun()
