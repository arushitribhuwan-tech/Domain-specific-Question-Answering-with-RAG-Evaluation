"""
Unit tests for Ingestion Module (cleaner, loader, chunker).
"""

try:
    import pytest
except ImportError:
    pytest = None
from src.ingestion.cleaner import TextCleaner
from src.ingestion.chunker import DocumentChunker
from src.ingestion.loader import DocumentLoader


def test_text_cleaner_whitespace():
    raw = "This is  a  test   with\n\n\n\nexcessive    spaces."
    cleaned = TextCleaner.clean(raw)
    assert "   " not in cleaned
    assert "\n\n\n" not in cleaned
    assert "This is a test with\n\nexcessive spaces." == cleaned


def test_text_cleaner_dividers_and_page_numbers():
    raw = (
        "Header Title\n"
        "====================================\n"
        "Page 1 of 10\n"
        "Main regulation body text.\n"
        "------------------------------------\n"
    )
    cleaned = TextCleaner.clean(raw)
    assert "====" not in cleaned
    assert "----" not in cleaned
    assert "Page 1 of 10" not in cleaned
    assert "Main regulation body text." in cleaned


def test_document_chunker_metadata():
    docs = [
        {
            "text": "Academic Regulation Section 1. Attendance must be 75 percent or higher. " * 30,
            "metadata": {
                "source_document": "sample_handbook.txt",
                "page_number": 2
            }
        }
    ]
    chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
    chunks = chunker.chunk_documents(docs)

    assert len(chunks) > 1
    first_chunk = chunks[0]
    assert "chunk_id" in first_chunk
    assert first_chunk["metadata"]["source_document"] == "sample_handbook.txt"
    assert first_chunk["metadata"]["page_number"] == 2
    assert "Attendance must be 75 percent" in first_chunk["text"]


def test_document_loader_graceful_missing_dir():
    loader = DocumentLoader(raw_data_dir="data/non_existent_folder_xyz")
    docs = loader.load_documents()
    assert docs == []
