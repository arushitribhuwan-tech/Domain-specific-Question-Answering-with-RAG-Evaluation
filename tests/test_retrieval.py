"""
Unit tests for Retrieval Module (embeddings, vector store, retriever).
"""

try:
    import pytest
except ImportError:
    pytest = None
from src.embeddings.embed import EmbeddingService, VectorStoreService
from src.retrieval.retriever import Retriever


def test_embedding_dimensions():
    embedder = EmbeddingService()
    vector = embedder.embed_text("What is the attendance requirement?")
    assert len(vector) == 384
    assert isinstance(vector[0], float)


def test_vector_indexing_and_similarity_search(tmp_path):
    embedder = EmbeddingService()
    vs = VectorStoreService(persist_dir=str(tmp_path / "chroma_test"), embedding_service=embedder)

    sample_chunks = [
        {
            "chunk_id": "chunk_1",
            "text": "Every student must maintain at least 75 percent attendance.",
            "metadata": {"source_document": "rules.txt", "page_number": 1}
        },
        {
            "chunk_id": "chunk_2",
            "text": "Tuition fee payment deadline is the 10th of every month.",
            "metadata": {"source_document": "finance.txt", "page_number": 3}
        }
    ]

    indexed_count = vs.index_chunks(sample_chunks)
    assert indexed_count == 2

    # Query for attendance
    results = vs.similarity_search("attendance requirement", k=1)
    assert len(results) >= 1
    assert "75 percent attendance" in results[0]["text"]


def test_retriever_formatting(tmp_path):
    embedder = EmbeddingService()
    vs = VectorStoreService(persist_dir=str(tmp_path / "chroma_test2"), embedding_service=embedder)
    vs.index_chunks([
        {
            "chunk_id": "chk_10",
            "text": "Passing grade requires 40 percent marks.",
            "metadata": {"source_document": "grading.txt", "page_number": 4}
        }
    ])

    retriever = Retriever(vectorstore_service=vs, top_k=1)
    retrieved = retriever.retrieve("passing grade marks")

    assert len(retrieved) == 1
    assert retrieved[0]["source_document"] == "grading.txt"
    assert retrieved[0]["page_number"] == 4
    assert "score" in retrieved[0]
