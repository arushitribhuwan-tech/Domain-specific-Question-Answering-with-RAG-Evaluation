# Project Directory Structure
## Domain-Specific Question Answering with RAG Evaluation

The following directory is designed specifically for this project and follows its actual workflow: document ingestion, cleaning & chunking, embedding & vector indexing, retrieval, grounded answer generation, and RAG evaluation.

```text
RootIQ-Domain-QA-RAG/
├── app.py                      # Main Streamlit chat UI
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── config.yaml                 # Chunk size, top-k, model settings
├── .gitignore                  # Git ignored files
│
├── data/
│   ├── raw/                    # Original domain PDFs / notices
│   ├── processed/              # Cleaned & chunked text
│   └── eval/
│       ├── test_qa.csv         # Test question-answer pairs
│       └── eval_results.csv    # RAGAS evaluation output
│
├── vectorstore/
│   └── chroma_db/              # Persisted ChromaDB vector index
│
├── src/
│   ├── ingestion/
│   │   ├── loader.py           # Load & extract text from documents
│   │   ├── cleaner.py          # Clean & normalize text
│   │   └── chunker.py          # Split text into chunks + metadata
│   ├── embeddings/
│   │   └── embed.py            # Generate embeddings (sentence-transformers)
│   ├── retrieval/
│   │   ├── retriever.py        # Similarity search / hybrid retrieval
│   │   └── reranker.py         # Optional cross-encoder re-ranking
│   ├── generation/
│   │   ├── prompt_builder.py   # Construct grounded prompt
│   │   └── llm_service.py      # Call local LLM via Ollama
│   ├── evaluation/
│   │   ├── run_eval.py         # Batch evaluation runner
│   │   └── metrics.py          # RAGAS metric wrappers
│   └── utils/
│       └── logger.py           # Logging utilities
│
├── notebooks/
│   └── exploration.ipynb       # EDA / prototyping notebook
│
├── logs/
│   ├── ingestion.log           # Ingestion run logs
│   └── eval.log                # Evaluation run logs
│
└── tests/
    ├── test_ingestion.py       # Unit tests for ingestion
    ├── test_retrieval.py       # Unit tests for retrieval
    └── test_generation.py      # Unit tests for generation
```
