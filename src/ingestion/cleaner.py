"""
Text cleaning and normalization module.
Removes boilerplate, extraneous headers/footers, excess whitespace, and control characters.
"""

import re
from src.utils.logger import get_logger

logger = get_logger("TextCleaner")


class TextCleaner:
    """Provides methods to clean raw document text before chunking."""

    @staticmethod
    def clean(text: str) -> str:
        """
        Cleans and normalizes raw text.

        Args:
            text: Raw input string extracted from PDF/TXT/HTML.

        Returns:
            Cleaned and normalized text string.
        """
        if not text or not isinstance(text, str):
            return ""

        # Remove non-printable / control characters (preserve newlines and tabs)
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

        # Remove decorative dividing lines (e.g. ==== or ---- or _____)
        cleaned = re.sub(r"^[=\-_*#]{3,}\s*$", "", cleaned, flags=re.MULTILINE)

        # Remove common page number patterns (e.g. 'Page 12 of 45', '--- 12 ---', 'Page 12')
        cleaned = re.sub(r"^\s*Page\s+\d+(\s+of\s+\d+)?\s*$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r"^\s*[-–—]\s*\d+\s*[-–—]\s*$", "", cleaned, flags=re.MULTILINE)

        # Normalize multiple spaces and tabs to single spaces
        cleaned = re.sub(r"[ \t]+", " ", cleaned)

        # Normalize 3 or more consecutive newlines to 2 newlines (preserve paragraph separation)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        # Strip leading and trailing whitespace
        return cleaned.strip()
