# 🎓 RootIQ-RAG: Complete Viva Presentation & Project Defense Cheatsheet
### Domain-Specific Question Answering with RAG Evaluation
**Author:** Arushi Tribhuwan • **Degree:** Final-Year Capstone Project

---

## ⏱️ Section 1: The 60-Second Elevator Pitch

> *"Good morning, esteemed examiners. Standard AI chatbots like ChatGPT fail in institutional environments because they hallucinate facts, lack up-to-date circulars, and provide zero verifiable source citations. In our project, **RootIQ-RAG**, we built a domain-specific question answering system using Retrieval-Augmented Generation. Our system ingests official college PDFs and circulars, chunks them into a persistent vector database using 384-dimensional embeddings, retrieves the top relevant evidence chunks for any student or faculty query, and generates grounded answers strictly backed by cited document names and page numbers. Furthermore, we quantitatively benchmarked our entire pipeline using industry-standard RAGAS metrics, achieving a **96.8% Faithfulness** and an end-to-end CPU query latency of under **2 seconds**, completely free and open-source."*

---

## 🎙️ Section 2: 5-Minute Presentation Script (Word-for-Word)

- **Minute 1: The Problem:** "Every semester, college administration offices are overwhelmed with repetitive student questions regarding exam eligibility, attendance condonation, credit requirements, and scholarships. Generic chatbots cannot be trusted because they fabricate policies. Our objective was to create an authoritative, grounded assistant that proves every claim with official page citations."
- **Minute 2: The Ingestion & Retrieval Pipeline:** "Our pipeline ingests multi-format documents (`.pdf`, `.txt`, `.html`). We clean noise using regex normalizers, chunk documents using recursive text splitting into 400-token windows with 50-token overlap, and embed them using `sentence-transformers/all-MiniLM-L6-v2`. These dense representations are persisted into a local ChromaDB collection indexed with Cosine Similarity."
- **Minute 3: Grounded Answer Generation:** "When a user asks a query, our Retriever extracts the top 5 most similar chunks. Our Prompt Builder wraps these verified chunks in rigid grounding instructions: the LLM is explicitly forbidden from using outside world knowledge and must state exact refusal if information is absent. Inference runs locally using Meta's `Llama 3.2 3B Instruct` via Ollama."
- **Minute 4: Quantitative RAGAS Evaluation:** "Rather than just building a demo UI, we evaluated the pipeline quantitatively against a curated 33-question benchmark across 5 dimensions: Context Precision (91.4%), Context Recall (89.2%), Faithfulness (96.8%), Answer Relevancy (88.5%), and Answer Correctness (87.1%)."
- **Minute 5: Live Demo & Conclusion:** "Our Streamlit UI offers an interactive chat with collapsible cited sources, a real-time evaluation dashboard, and dynamic document upload capabilities. The entire project runs 100% locally on standard laptops at zero cloud cost."

---

## 📐 Section 3: Mathematical & Algorithmic Foundations

### 1. Cosine Similarity (Vector Distance Metric)
Retrieval ranks chunks based on the cosine angle between the query vector $\vec{q}$ and document chunk vector $\vec{d}$:
$$\text{Cosine Similarity}(\vec{q}, \vec{d}) = \frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\| \|\vec{d}\|} = \frac{\sum_{i=1}^{n} q_i d_i}{\sqrt{\sum_{i=1}^{n} q_i^2} \sqrt{\sum_{i=1}^{n} d_i^2}}$$
In ChromaDB, distance is measured as: $\text{Cosine Distance} = 1 - \text{Cosine Similarity}$.

### 2. RAGAS Evaluation Metric Formulas
- **Faithfulness:** Measures factual grounding (detects hallucination):
  $$\text{Faithfulness} = \frac{|\text{Number of Claims in Answer supported by Context}|}{|\text{Total Claims in Answer}|}$$
- **Context Precision:** Evaluates if relevant chunks are ranked at the top:
  $$\text{Context Precision@k} = \frac{\sum_{k=1}^{K} (\text{Precision@k} \times v_k)}{\text{Total Relevant Chunks in Top-k}}$$
- **Context Recall:** Evaluates retrieval completeness against ground truth:
  $$\text{Context Recall} = \frac{|\text{Ground-Truth Sentences Attributed to Retrieved Chunks}|}{|\text{Total Sentences in Ground-Truth Reference}|}$$
- **Answer Correctness (Semantic F1):** Weighted harmonic mean of precision and recall between generated and reference answers:
  $$\text{F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## ❓ Section 4: Top 15 Tough Viva Questions & Examiner Answers

#### Q1: Why did you choose RAG instead of Fine-Tuning an LLM?
**Answer:** Fine-tuning modifies model weights, which is computationally expensive, prone to catastrophic forgetting, still hallucinates, and requires complete retraining every time an academic regulation changes. RAG separates knowledge storage from reasoning: updating a regulation takes seconds by simply dropping a new PDF into the vector store with zero retraining cost and transparent citations.

#### Q2: Why did you choose `all-MiniLM-L6-v2` as your embedding model?
**Answer:** `all-MiniLM-L6-v2` produces compact 384-dimensional embeddings with only 22 million parameters. It computes embeddings in less than 15 milliseconds on a CPU while outperforming much larger legacy models on the MTEB retrieval benchmark.

#### Q3: What is the difference between Token Chunking and Character Chunking?
**Answer:** Character chunking can split words or sentences abruptly. We use LangChain's `RecursiveCharacterTextSplitter`, which respects natural linguistic boundaries (paragraphs `\n\n`, sentences `. `, and words) while maintaining a target size of ~400 tokens (~1600 characters) with a 50-token overlap to preserve semantic context across chunk edges.

#### Q4: What happens if a student asks an out-of-domain question (e.g., "What is the recipe for cake?")?
**Answer:** The retriever returns low-similarity chunks (similarity score < 0.20). The prompt and guardrail logic detect insufficient evidence and trigger our strict refusal response: *"I do not have enough information in the provided domain documents to answer this question."* This prevents hallucination.

#### Q5: What is ChromaDB and why is it used instead of a relational database like MySQL?
**Answer:** Traditional relational databases search via exact string matches (SQL `LIKE`). ChromaDB is an Approximate Nearest Neighbor (ANN) vector database using Hierarchical Navigable Small World (HNSW) graphs to search by semantic meaning in high-dimensional vector space in logarithmic time $\mathcal{O}(\log N)$.

#### Q6: What does the Faithfulness metric measure in RAGAS?
**Answer:** Faithfulness measures whether every claim made in the generated answer can be mathematically traced back to the retrieved context chunks. A score of 1.0 indicates 0% hallucination.

#### Q7: What is the difference between Context Precision and Context Recall?
**Answer:** Context Precision measures signal-to-noise ratio: how many of the top retrieved chunks are actually useful. Context Recall measures coverage: did the retriever successfully fetch all pieces of information required to answer the question?

#### Q8: How does Ollama run Llama 3.2 3B on a standard laptop without a GPU?
**Answer:** Ollama uses `llama.cpp` with 4-bit quantization (GGUF format), reducing model weights from 16-bit floating point to 4-bit integers. This shrinks memory footprint from ~8GB down to ~2GB, enabling smooth CPU inference.

#### Q9: What is the purpose of chunk overlap?
**Answer:** If an important rule or sentence is split exactly at the chunk boundary, both chunks might lose crucial context. Overlap (50 tokens) guarantees that boundary sentences appear in full across adjacent chunks.

#### Q10: Why is temperature set to 0.1?
**Answer:** In creative writing, higher temperatures (0.7–1.0) encourage diversity. In regulatory and legal domain Q&A, we set temperature to 0.1 to enforce deterministic, strictly factual, and conservative generation.

#### Q11: Can this system handle scanned image PDFs?
**Answer:** Our MVP ingestion pipeline uses text-layer extractors (`pypdf` and `PyMuPDF`). For scanned documents without text layers, an OCR pre-processing step (such as `Tesseract` or `easyocr`) would be required before feeding text to the cleaner.

#### Q12: How does the system scale if the college adds 500 documents?
**Answer:** ChromaDB uses HNSW indexing, which scales sub-linearly with data size. Even with 10,000 chunks, vector similarity search takes less than 50 milliseconds.

#### Q13: What is Hybrid Retrieval?
**Answer:** Hybrid retrieval combines dense semantic vector search with sparse keyword search (BM25) using Reciprocal Rank Fusion (RRF). It ensures acronyms, course codes (e.g., "CS-301"), and specific regulation numbers are matched accurately alongside conceptual queries.

#### Q14: What is the purpose of the test_qa.csv file?
**Answer:** It acts as the ground-truth benchmark containing 33 pairs of questions and human-verified reference answers, allowing automated, repeatable evaluation across pipeline changes.

#### Q15: What are the future enhancements for this project?
**Answer:** 1) Multilingual support for regional languages using multilingual embedding models; 2) Conversational multi-turn memory; 3) Role-based access control for administrative vs. student documents.
