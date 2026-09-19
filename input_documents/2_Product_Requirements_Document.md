# Product Requirements Document (PRD)
## Domain-Specific Question Answering with RAG Evaluation
### Retrieval-Augmented Generation for Reliable, Source-Grounded Domain Q&A • TY College Project

---

### 1. Problem Statement & Objective

#### Problem
General-purpose chatbots often give vague, outdated, or hallucinated answers when asked detailed questions about a specific domain (e.g., university admissions, academic regulations, scholarships, course information). There is no easy way for students/staff to get answers that are grounded in verified official documents and backed by visible source citations.

#### Objective
- Build a domain-specific Question Answering assistant using Retrieval-Augmented Generation (RAG).
- Ground every generated answer in retrieved chunks from trusted domain documents, with citations.
- Quantitatively evaluate the RAG pipeline itself (retrieval quality + answer quality + latency), not just ship a chatbot.
- Keep the entire stack free, open-source, and runnable on a single laptop.

---

### 2. Target Users
- **Students:** Quick, trustworthy answers about admissions, exams, syllabus, scholarships, fees, and deadlines.
- **Faculty / Administrative staff:** Reduce repetitive queries by pointing users to an assistant backed by official circulars/handbooks.
- **Prospective applicants:** Self-service Q&A on eligibility, courses, and procedures without navigating scattered PDFs.
- **Evaluators / project reviewers:** A measurable, benchmarked system rather than a black-box chatbot demo.

---

### 3. Key Features / Functional Requirements
- **Document ingestion pipeline:** Upload/parse PDFs, notices, and web pages; clean and chunk text with metadata (doc name, section, page).
- **Semantic retrieval:** Embed chunks into a vector store and retrieve top-k relevant chunks per question.
- **Grounded answer generation:** LLM answers strictly from retrieved context; refuses/flags when evidence is insufficient.
- **Source citation display:** Every answer shows the document name, section/page it was drawn from.
- **Chat-style Q&A interface:** Simple web UI to ask questions and view answers with sources.
- **Evaluation module:** Run a prepared QA test set through the pipeline and compute RAG metrics automatically.
- **Configurable settings:** Swap chunk size, top-k, embedding model, or LLM to compare performance.
- **Admin/upload panel:** Add or refresh domain documents without touching code.

---

### 4. System Workflow & Pipeline Steps
$$\text{User Question} \rightarrow \text{Query Embedding} \rightarrow \text{Vector DB Similarity Search} \rightarrow \text{Top-k Relevant Chunks} \rightarrow \text{Prompt Construction} \rightarrow \text{LLM Generation} \rightarrow \text{Grounded Answer + Citations} \rightarrow \text{Evaluation Log}$$

1. **Step 1 — Domain & Dataset Selection:** Pick one domain (e.g., academic regulations) and gather reliable PDFs/notices/web pages.
2. **Step 2 — Document Processing:** Extract text, remove noise, split into overlapping chunks (~300–500 tokens), attach metadata.
3. **Step 3 — Embedding & Indexing:** Convert chunks into vectors with an embedding model; store in a local vector database.
4. **Step 4 — Retrieval:** Embed the user question and run similarity search (optionally + keyword search) to fetch top-k chunks.
5. **Step 5 — Generation:** Pass question + retrieved context to the LLM in a strict "answer only from context" prompt; return answer with citations.
6. **Step 6 — Evaluation:** Run a held-out QA set through the pipeline; score context precision/recall, faithfulness, answer relevance/correctness, and latency.

---

### 5. Recommended Open-Source Tech Stack
- **Frontend / UI:** Streamlit (chat interface, source panel, evaluation dashboard)
- **Backend / Orchestration:** Python 3.10+, LangChain (loaders, splitters, RAG chain)
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (fast, 384-dim, local)
- **Vector Database:** ChromaDB (local, file-based, zero setup)
- **Local LLM Runtime:** Ollama running `Llama 3.2 3B` / `Llama 3.1 8B Instruct`
- **Evaluation Engine:** RAGAS (Context Precision, Context Recall, Faithfulness, Relevancy, Correctness) + pandas
- **Deployment:** Local machine; $0 cost, completely open-source

---

### 6. Evaluation Metrics
- **Context Precision:** How many of the retrieved chunks are actually relevant to the question.
- **Context Recall:** Whether the retrieved chunks contain the information needed to answer.
- **Faithfulness:** Whether the generated answer is actually supported by the retrieved context (no hallucination).
- **Answer Relevance:** Whether the answer directly addresses what was asked.
- **Answer Correctness:** How closely the generated answer matches a human reference answer.
- **Latency:** End-to-end time for retrieval + generation per query.
