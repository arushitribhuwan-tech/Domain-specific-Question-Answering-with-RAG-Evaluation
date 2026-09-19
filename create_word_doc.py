"""
Script to generate a professional, complete Word document (.docx)
containing all project documentation, architecture, and complete source code.
"""

import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn


def set_cell_background(cell, fill_hex):
    """Sets background color of a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def add_code_block(doc, code_text):
    """Adds a styled code block with light gray background and border."""
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    cell = table.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, "F4F5F7")

    # Set margins inside cell
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="120" w:type="dxa"/>'
        f'<w:left w:w="180" w:type="dxa"/>'
        f'<w:bottom w:w="120" w:type="dxa"/>'
        f'<w:right w:w="180" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.05
    run = p.add_run(code_text)
    run.font.name = "Consolas"
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor(30, 41, 59)

    doc.add_paragraph()  # Spacing after table


def build_docx(project_root: str, output_docx_path: str):
    doc = Document()

    # Page Margins
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    # ------------------- TITLE / COVER -------------------
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_p.add_run("Domain-Specific Question Answering with RAG Evaluation\n(RootIQ-Domain-QA-RAG)")
    title_run.font.name = "Calibri"
    title_run.font.size = Pt(22)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(24, 76, 148)

    sub_p = doc.add_paragraph()
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_run = sub_p.add_run("TY College Capstone / Final Year Project Report & Complete Source Code\n")
    sub_run.font.size = Pt(13)
    sub_run.font.italic = True
    sub_run.font.color.rgb = RGBColor(100, 116, 139)

    meta_p = doc.add_paragraph()
    meta_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta_p.add_run(
        "Domain: Academic Regulations & Institutional Policies\n"
        "Tech Stack: Python 3.10+, Streamlit, LangChain, ChromaDB, Sentence-Transformers, Ollama, RAGAS\n"
        "Architecture: 5-Layer Modular RAG Pipeline (Ingestion, Storage, Retrieval, Generation, Evaluation)\n"
    )

    doc.add_heading("1. Project Executive Summary", level=1)
    doc.add_paragraph(
        "General-purpose Large Language Models (LLMs) frequently hallucinate and lack specific domain awareness "
        "when answering detailed queries regarding college admissions, grading policies, attendance regulations, "
        "scholarships, and examination rules.\n\n"
        "This project implements a complete, end-to-end, and production-ready Retrieval-Augmented Generation (RAG) "
        "system that strictly grounds its responses in verified domain documents with citation tracking (document name, "
        "page number, and similarity score). Furthermore, the pipeline incorporates an automated benchmarking framework "
        "measuring 5 core RAGAS metrics (Context Precision, Context Recall, Faithfulness, Answer Relevancy, and Answer Correctness) "
        "alongside sub-stage latency profiling."
    )

    doc.add_heading("2. System Architecture & Workflow", level=1)
    arch_text = (
        "High-Level Pipeline Workflow:\n\n"
        "1. Ingestion: Raw documents (.pdf, .txt, .html) -> Clean headers/footers/whitespace -> "
        "Split into 400-token chunks with 50-token overlap -> Attach chunk_id, source_doc, page_num.\n"
        "2. Embedding & Indexing: Convert chunks to 384-dim dense vectors using sentence-transformers/all-MiniLM-L6-v2 -> "
        "Persist in ChromaDB collection.\n"
        "3. Retrieval: User query -> Query embedding -> Cosine similarity search (top-k=5) -> "
        "Optional BM25 hybrid ranking & Cross-Encoder re-ranking.\n"
        "4. Grounded Generation: Build strict prompt with numbered source blocks -> Invoke local LLM (Ollama Llama 3.2 3B) -> "
        "Return factual answer with source citations (or refusal for out-of-domain queries).\n"
        "5. Quantitative Evaluation: Run held-out test_qa.csv -> Measure retrieval & generation latency -> "
        "Compute 5 RAGAS metrics -> Export eval_results.csv."
    )
    add_code_block(doc, arch_text)

    doc.add_heading("3. Project Directory Structure", level=1)
    tree_text = (
        "RootIQ-Domain-QA-RAG/\n"
        "├── app.py                          # Streamlit web UI (Chatbot + Eval Dashboard + Ingestion)\n"
        "├── requirements.txt                # Python dependencies\n"
        "├── README.md                       # Comprehensive setup & project documentation\n"
        "├── config.yaml                     # Externalized configuration (chunk size, models, paths)\n"
        "├── .gitignore                      # Git ignored files\n"
        "├── run_tests.py                    # Standalone test runner (pytest / unittest)\n"
        "│\n"
        "├── data/\n"
        "│   ├── raw/                        # Domain source documents (PDFs, TXT notices, handbooks)\n"
        "│   │   └── academic_regulations_handbook.txt\n"
        "│   ├── processed/                  # Cleaned and chunked metadata (chunks_summary.json)\n"
        "│   └── eval/\n"
        "│       ├── test_qa.csv             # 32 Curated ground-truth test QA pairs\n"
        "│       └── eval_results.csv        # Evaluation results with RAGAS metrics & latency\n"
        "│\n"
        "├── vectorstore/\n"
        "│   └── chroma_db/                  # Persisted ChromaDB vector database index\n"
        "│\n"
        "├── src/\n"
        "│   ├── __init__.py\n"
        "│   ├── ingestion/\n"
        "│   │   ├── __init__.py\n"
        "│   │   ├── loader.py               # Document loader (PDF/TXT/MD/HTML) with error handling\n"
        "│   │   ├── cleaner.py              # Text cleaning and normalization\n"
        "│   │   └── chunker.py              # Recursive character text chunking with metadata\n"
        "│   ├── embeddings/\n"
        "│   │   ├── __init__.py\n"
        "│   │   └── embed.py                # Embedding wrapper & ChromaDB vectorstore service\n"
        "│   ├── retrieval/\n"
        "│   │   ├── __init__.py\n"
        "│   │   ├── retriever.py            # Cosine similarity search + BM25 hybrid retrieval\n"
        "│   │   └── reranker.py             # Optional Cross-Encoder re-ranking\n"
        "│   ├── generation/\n"
        "│   │   ├── __init__.py\n"
        "│   │   ├── prompt_builder.py       # Grounded prompt builder enforcing strict evidence\n"
        "│   │   └── llm_service.py          # Local Ollama REST client with offline fallback\n"
        "│   ├── evaluation/\n"
        "│   │   ├── __init__.py\n"
        "│   │   ├── run_eval.py             # Batch evaluation runner & latency profiler\n"
        "│   │   └── metrics.py              # RAGAS metrics (precision, recall, faithfulness, etc.)\n"
        "│   └── utils/\n"
        "│       ├── __init__.py\n"
        "│       └── logger.py               # Rotating file and console logger\n"
        "│\n"
        "├── notebooks/\n"
        "│   └── exploration.ipynb           # EDA and prototyping Jupyter notebook\n"
        "│\n"
        "├── logs/\n"
        "│   ├── ingestion.log               # Ingestion execution logs\n"
        "│   └── eval.log                    # Evaluation run logs\n"
        "│\n"
        "└── tests/\n"
        "    ├── __init__.py\n"
        "    ├── test_ingestion.py           # Unit tests for cleaner, loader, and chunker\n"
        "    ├── test_retrieval.py           # Unit tests for vector indexing and retriever\n"
        "    └── test_generation.py          # Unit tests for prompt builder, refusals, and metrics"
    )
    add_code_block(doc, tree_text)

    # ------------------- SOURCE CODE SECTIONS -------------------
    doc.add_heading("4. Complete Source Code (File by File)", level=1)

    files_to_include = [
        ("Configuration File", "config.yaml", "Externalized parameters for chunking, models, paths, and retrieval settings."),
        ("Python Dependencies", "requirements.txt", "Pinned library dependencies for reproducible deployment."),
        ("Git Ignore", ".gitignore", "Version control exclusions for virtual environments, vector databases, and logs."),
        ("Standalone Test Runner", "run_tests.py", "Test runner supporting both pytest and standard library execution."),
        ("Sample Domain Handbook", "data/raw/academic_regulations_handbook.txt", "Comprehensive verified institutional academic regulations document."),
        ("Curated Test QA Dataset", "data/eval/test_qa.csv", "32 Curated ground-truth Q&A pairs for automated benchmarking."),
        ("Package Init - Root", "src/__init__.py", "Root package initialization."),
        ("Package Init - Utils", "src/utils/__init__.py", "Utilities package initialization."),
        ("Logging Utility", "src/utils/logger.py", "Centralized console and rotating file logging handler."),
        ("Package Init - Ingestion", "src/ingestion/__init__.py", "Ingestion module exports."),
        ("Text Cleaner", "src/ingestion/cleaner.py", "Normalization of whitespace, boilerplate, page numbers, and dividers."),
        ("Document Loader", "src/ingestion/loader.py", "Multi-format document loading (PDF, TXT, MD, HTML) with error skipping."),
        ("Document Chunker", "src/ingestion/chunker.py", "Recursive text splitting with chunk_id, source_document, and page_number attachment."),
        ("Package Init - Embeddings", "src/embeddings/__init__.py", "Embeddings module exports."),
        ("Embeddings & Vectorstore Service", "src/embeddings/embed.py", "SentenceTransformer 384-dim encoding and persistent ChromaDB index management."),
        ("Package Init - Retrieval", "src/retrieval/__init__.py", "Retrieval module exports."),
        ("Re-Ranker Module", "src/retrieval/reranker.py", "Cross-Encoder semantic scoring and lexical boost re-ranking."),
        ("Retriever Module", "src/retrieval/retriever.py", "Similarity search, candidate filtering, and metadata formatting."),
        ("Package Init - Generation", "src/generation/__init__.py", "Generation module exports."),
        ("Prompt Builder", "src/generation/prompt_builder.py", "Grounded prompt construction, evidence restrictions, and refusal rules."),
        ("LLM Service Client", "src/generation/llm_service.py", "Ollama REST API client with local offline extractive fallback and latency profiling."),
        ("Package Init - Evaluation", "src/evaluation/__init__.py", "Evaluation module exports."),
        ("RAGAS Evaluation Metrics", "src/evaluation/metrics.py", "Precision, Recall, Faithfulness, Relevancy, and Correctness implementations."),
        ("Batch Evaluation Runner", "src/evaluation/run_eval.py", "End-to-end benchmark execution across test_qa.csv with CSV report export."),
        ("Streamlit Web Application", "app.py", "Full multi-tab web application for Chat, Evaluation Dashboard, and Document Ingestion."),
        ("Package Init - Tests", "tests/__init__.py", "Test package initialization."),
        ("Unit Tests - Ingestion", "tests/test_ingestion.py", "Unit tests for cleaner, loader, and chunker."),
        ("Unit Tests - Retrieval", "tests/test_retrieval.py", "Unit tests for embedding dimensions, ChromaDB indexing, and similarity search."),
        ("Unit Tests - Generation", "tests/test_generation.py", "Unit tests for prompt construction, out-of-domain refusals, and metric calculation.")
    ]

    for title, rel_path, desc in files_to_include:
        full_path = os.path.join(project_root, rel_path)
        doc.add_heading(f"{title}: {rel_path}", level=2)
        doc.add_paragraph(f"Description: {desc}\nRelative Path: {rel_path}")

        if os.path.exists(full_path):
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                code_content = f.read()
            add_code_block(doc, code_content)
        else:
            doc.add_paragraph(f"[File not found at: {full_path}]")

    # ------------------- HOW TO RUN -------------------
    doc.add_heading("5. Installation & Execution Guide", level=1)
    run_instructions = (
        "Step 1: Open PowerShell / Terminal and Navigate to Project Directory\n"
        "cd C:\\Users\\Tribhuwan\\.gemini\\antigravity\\scratch\\RootIQ-Domain-QA-RAG\n\n"
        "Step 2: Create & Activate Virtual Environment\n"
        "python -m venv venv\n"
        ".\\venv\\Scripts\\Activate.ps1\n\n"
        "Step 3: Install Required Dependencies\n"
        "pip install -r requirements.txt\n\n"
        "Step 4: Execute Unit Test Suite\n"
        "python run_tests.py\n"
        "(Output: 10 passed, 0 failed)\n\n"
        "Step 5: Run Automated RAGAS Batch Evaluation\n"
        "python -m src.evaluation.run_eval\n"
        "(Evaluates all 32 curated questions, records latencies, and outputs eval_results.csv)\n\n"
        "Step 6: Launch Interactive Streamlit Web UI\n"
        "streamlit run app.py\n"
        "(Opens web interface at http://localhost:8501 featuring Chat, Metrics Dashboard, and Document Ingestion)"
    )
    add_code_block(doc, run_instructions)

    doc.save(output_docx_path)
    print(f"Successfully created Word document at: {output_docx_path}")


if __name__ == "__main__":
    project_dir = r"C:\Users\Tribhuwan\.gemini\antigravity\scratch\RootIQ-Domain-QA-RAG"
    output_path = os.path.join(project_dir, "Domain_Specific_QA_RAG_Project_Documentation.docx")
    build_docx(project_dir, output_path)
