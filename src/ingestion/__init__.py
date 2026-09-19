"""
Document Ingestion & Processing Module.
Includes loading, text cleaning, and chunking with metadata attachment.
"""

from src.ingestion.loader import DocumentLoader
from src.ingestion.cleaner import TextCleaner
from src.ingestion.chunker import DocumentChunker

__all__ = ["DocumentLoader", "TextCleaner", "DocumentChunker"]
