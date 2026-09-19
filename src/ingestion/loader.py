"""
Document loader module.
Loads raw documents (PDF, TXT, MD, HTML) from the specified directory
and attaches initial metadata (source_document, page_number).
Gracefully handles unreadable or corrupted files.
"""

import os
from typing import List, Dict, Any
from src.utils.logger import get_logger

logger = get_logger("DocumentLoader", "logs/ingestion.log")


class DocumentLoader:
    """Loads source domain documents from a folder into standardized document dicts."""

    def __init__(self, raw_data_dir: str = "data/raw"):
        self.raw_data_dir = raw_data_dir

    def load_documents(self) -> List[Dict[str, Any]]:
        """
        Loads all supported documents from raw_data_dir.

        Returns:
            List of dicts: [{'text': str, 'metadata': {'source_document': str, 'page_number': int}}]
        """
        if not os.path.exists(self.raw_data_dir):
            logger.warning(f"Raw data directory '{self.raw_data_dir}' does not exist. Creating it.")
            os.makedirs(self.raw_data_dir, exist_ok=True)
            return []

        loaded_documents: List[Dict[str, Any]] = []
        files = [f for f in os.listdir(self.raw_data_dir) if os.path.isfile(os.path.join(self.raw_data_dir, f))]

        if not files:
            logger.warning(f"No files found in '{self.raw_data_dir}'.")
            return []

        for filename in files:
            file_path = os.path.join(self.raw_data_dir, filename)
            ext = os.path.splitext(filename)[1].lower()

            try:
                if ext == ".pdf":
                    docs = self._load_pdf(file_path, filename)
                elif ext in [".txt", ".md"]:
                    docs = self._load_text(file_path, filename)
                elif ext in [".html", ".htm"]:
                    docs = self._load_html(file_path, filename)
                else:
                    logger.info(f"Skipping unsupported file format: {filename}")
                    continue

                loaded_documents.extend(docs)
                logger.info(f"Successfully loaded '{filename}' ({len(docs)} pages/sections).")

            except Exception as e:
                # FRD / TRD requirement: log and skip unreadable/corrupt files without crashing
                logger.error(f"Failed to load '{filename}': {str(e)}. Skipping file.")
                continue

        logger.info(f"Ingested total of {len(loaded_documents)} document sections from {len(files)} files.")
        return loaded_documents

    def _load_pdf(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        """Extracts text page-by-page from a PDF using pypdf or PyMuPDF."""
        docs = []

        # Try pypdf first
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append({
                        "text": text,
                        "metadata": {
                            "source_document": filename,
                            "page_number": page_idx + 1
                        }
                    })
            if docs:
                return docs
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"pypdf encountered issue with {filename}: {e}. Trying PyMuPDF fallback.")

        # PyMuPDF fallback
        try:
            import fitz
            doc = fitz.open(file_path)
            for page_idx, page in enumerate(doc):
                text = page.get_text() or ""
                if text.strip():
                    docs.append({
                        "text": text,
                        "metadata": {
                            "source_document": filename,
                            "page_number": page_idx + 1
                        }
                    })
            return docs
        except ImportError:
            logger.error("Neither 'pypdf' nor 'PyMuPDF' is installed. Please install them to parse PDFs.")
            return []

    def _load_text(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        """Reads plain text or markdown files."""
        encodings = ["utf-8", "latin-1", "cp1252"]
        content = ""
        for enc in encodings:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue

        if not content.strip():
            return []

        # Return single page document (or split by standard section dividers if present)
        return [{
            "text": content,
            "metadata": {
                "source_document": filename,
                "page_number": 1
            }
        }]

    def _load_html(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        """Extracts plain text from HTML files."""
        encodings = ["utf-8", "latin-1"]
        content = ""
        for enc in encodings:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue

        # Simple HTML tag stripper
        import re
        clean_text = re.sub(r"<[^>]+>", " ", content)
        return [{
            "text": clean_text,
            "metadata": {
                "source_document": filename,
                "page_number": 1
            }
        }]
