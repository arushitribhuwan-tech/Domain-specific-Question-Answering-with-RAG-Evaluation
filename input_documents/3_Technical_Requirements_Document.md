# Technical Requirements Document (TRD)
## Domain-Specific Question Answering with RAG Evaluation
**Version 1.0 • TY College Project**

---

### 1. Purpose & Scope
This document specifies the technical architecture, components, data design, interfaces, and environment needed to build and run the Domain-Specific Question Answering (RAG) system. It defines how the system is technically implemented.

---

### 2. System Architecture

#### 2.1 High-Level Architecture
The system follows a modular RAG pipeline architecture with five layers: Ingestion, Storage, Retrieval, Generation, and Evaluation, fronted by a Streamlit UI.

```
[Documents]
    │
    ▼
[Ingestion Module] ──► [Embedding Model] ──► [Vector DB: ChromaDB]
                                                   │
[User] ──► [Streamlit UI]                          │
               │                                   │
               ▼                                   ▼
       [Query Handler] ──────────────────► [Retriever Module]
                                                   │
                                                   ▼
       [Prompt Builder] ◄────────────────── [Top-k Chunks]
               │
               ▼
      [LLM Service: Ollama]
               │
               ├─────────────────────────► [Answer + Citations] ──► [UI]
               │
               ▼
       [Eval Logger] ────────────────────► [RAGAS Evaluation]
```

---

### 3. Component-Level Technical Design

#### 3.1 Ingestion Module
- **Input:** Raw files (`.pdf`, `.txt`, `.html`) placed in `/data/raw` directory.
- **Process:** Extract text $\rightarrow$ clean (remove headers/footers/whitespace) $\rightarrow$ split into chunks (`chunk_size=400` tokens, `overlap=50`) using `RecursiveCharacterTextSplitter` $\rightarrow$ attach metadata (`source`, `page`, `chunk_id`).
- **Output:** List of chunk objects `{id, text, metadata}` written to vector store.

#### 3.2 Embedding & Vector Store
- **Model:** `sentence-transformers/all-MiniLM-L6-v2`, output dimension 384.
- **Storage:** ChromaDB persistent collection at `/vectorstore/chroma_db`; one collection per domain.
- **Indexing:** Cosine similarity search; supports metadata filtering (e.g., by document name).

#### 3.3 Retrieval Module
- **Input:** User query string.
- **Process:** Embed query with the same embedding model $\rightarrow$ run `similarity_search(query, k=5)` against ChromaDB $\rightarrow$ optionally apply BM25 hybrid scoring and cross-encoder re-ranking.
- **Output:** Ranked list of top-k chunks with similarity scores and metadata.

#### 3.4 Generation Module
- **Input:** User query + top-k retrieved chunks.
- **Prompt Template:** System instruction restricting the LLM to provided context, followed by numbered context chunks and the question.
- **Model Invocation:** Local call to Ollama's REST API (`http://localhost:11434`) with the configured model name (`llama3.2:3b`).
- **Output:** Generated answer text + list of cited chunk IDs/sources.

#### 3.5 Evaluation Module
- **Input:** Test QA set (CSV/JSON of question + reference answer).
- **Process:** Run each question through the full pipeline, collect `{question, contexts, answer, reference}` $\rightarrow$ pass to RAGAS metric functions (`context_precision`, `context_recall`, `faithfulness`, `answer_relevancy`, `answer_correctness`) $\rightarrow$ record latency per stage.
- **Output:** A results table (`eval_results.csv`) and summary statistics rendered in the UI or exported.

#### 3.6 UI Layer
- **Framework:** Streamlit single-page app with a chat component (`st.chat_message`) and an expandable sources panel (`st.expander`).

---

### 4. Data Design

#### 4.1 Chunk Metadata Schema
| Field | Type | Description |
| :--- | :--- | :--- |
| `chunk_id` | string | Unique identifier for the chunk |
| `source_document` | string | Original file name |
| `page_number` | integer | Page or section number in the source document |
| `text` | string | The chunk's raw text content |
| `embedding` | float[384] | Vector representation of the chunk |

#### 4.2 Evaluation Log Schema
| Field | Type | Description |
| :--- | :--- | :--- |
| `run_id` | string | Identifier for a specific evaluation run |
| `question` | string | Test question |
| `retrieved_chunks` | list[string] | Chunk IDs retrieved for this question |
| `generated_answer` | string | Answer produced by the LLM |
| `reference_answer` | string | Human-verified reference answer |
| `metrics` | JSON | Scores for context precision, recall, faithfulness, relevancy, correctness |
| `latency_ms` | integer | End-to-end response time |

---

### 5. Interface Specifications
- **Internal API:** `retrieve(query: str, k: int) -> List[Chunk]`: returns ranked chunks for a query.
- **Internal API:** `generate_answer(query: str, chunks: List[Chunk]) -> AnswerResult`: returns `{answer, sources}`.
- **Internal API:** `evaluate(test_set_path: str) -> DataFrame`: runs batch evaluation and returns a metrics table.
- **External Call:** Ollama REST API: `POST /api/generate` with `{model, prompt}` to obtain the LLM's completion.
