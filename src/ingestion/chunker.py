"""
Document chunker module.
Splits cleaned documents into overlapping chunks with metadata (chunk_id, source_document, page_number).
Exports processed chunks to data/processed/ for transparency and inspection.
"""

import os
import json
import uuid
from typing import List, Dict, Any
from src.ingestion.cleaner import TextCleaner
from src.utils.logger import get_logger

logger = get_logger("DocumentChunker", "logs/ingestion.log")


class DocumentChunker:
    """Splits documents into overlapping chunks using character-level/token recursive splitting."""

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50, processed_data_dir: str = "data/processed"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.processed_data_dir = processed_data_dir
        self.cleaner = TextCleaner()

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Takes raw document dicts, cleans their text, and splits them into chunks with metadata.

        Args:
            documents: List of dicts with 'text' and 'metadata'.

        Returns:
            List of chunk dicts:
            [
                {
                    'chunk_id': str,
                    'text': str,
                    'metadata': {
                        'chunk_id': str,
                        'source_document': str,
                        'page_number': int
                    }
                }
            ]
        """
        all_chunks: List[Dict[str, Any]] = []

        # Check if LangChain's RecursiveCharacterTextSplitter is available
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            # Character approximation: ~4 chars per token -> chunk_size * 4 characters
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size * 4,
                chunk_overlap=self.chunk_overlap * 4,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            use_langchain = True
        except ImportError:
            use_langchain = False
            logger.info("LangChain not installed. Using built-in recursive chunker.")

        chunk_counter = 0

        for doc in documents:
            raw_text = doc.get("text", "")
            cleaned_text = self.cleaner.clean(raw_text)

            if not cleaned_text:
                continue

            metadata = doc.get("metadata", {})
            source_doc = metadata.get("source_document", "unknown_doc")
            page_num = metadata.get("page_number", 1)

            if use_langchain:
                splits = splitter.split_text(cleaned_text)
            else:
                splits = self._fallback_split_text(cleaned_text, self.chunk_size * 4, self.chunk_overlap * 4)

            for idx, split_text in enumerate(splits):
                if not split_text.strip():
                    continue

                chunk_counter += 1
                chunk_id = f"{source_doc}_p{page_num}_c{idx+1}"
                chunk_item = {
                    "chunk_id": chunk_id,
                    "text": split_text.strip(),
                    "metadata": {
                        "chunk_id": chunk_id,
                        "source_document": source_doc,
                        "page_number": page_num
                    }
                }
                all_chunks.append(chunk_item)

        logger.info(f"Chunked {len(documents)} documents into {len(all_chunks)} chunks.")

        # Save processed chunks to data/processed/ for tracking & EDA
        self._save_processed_chunks(all_chunks)

        return all_chunks

    def _fallback_split_text(self, text: str, max_chars: int, overlap_chars: int) -> List[str]:
        """Simple deterministic sliding-window character text splitter with paragraph boundary awareness."""
        if len(text) <= max_chars:
            return [text]

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + max_chars
            if end >= text_len:
                chunks.append(text[start:].strip())
                break

            # Try to break on paragraph or sentence
            newline_pos = text.rfind("\n", start, end)
            period_pos = text.rfind(". ", start, end)

            if newline_pos > start + (max_chars // 2):
                split_point = newline_pos + 1
            elif period_pos > start + (max_chars // 2):
                split_point = period_pos + 1
            else:
                space_pos = text.rfind(" ", start, end)
                split_point = space_pos + 1 if space_pos > start else end

            chunks.append(text[start:split_point].strip())
            start = max(start + 1, split_point - overlap_chars)

        return [c for c in chunks if c]

    def _save_processed_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Saves metadata of generated chunks to processed_data_dir."""
        try:
            os.makedirs(self.processed_data_dir, exist_ok=True)
            output_file = os.path.join(self.processed_data_dir, "chunks_summary.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved chunks summary to '{output_file}'.")
        except Exception as e:
            logger.error(f"Failed to save processed chunks summary: {e}")
