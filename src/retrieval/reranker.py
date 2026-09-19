"""
Optional Cross-Encoder Re-ranker module.
Re-ranks top-k retrieved chunks to prioritize true semantic relevance.
"""

from typing import List, Dict, Any
from src.utils.logger import get_logger

logger = get_logger("ReRanker")


class ReRanker:
    """Re-ranks retrieved candidates using a cross-encoder model or lexical match booster."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", enabled: bool = False):
        self.model_name = model_name
        self.enabled = enabled
        self._model = None
        if self.enabled:
            self._init_model()

    def _init_model(self):
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            logger.info(f"Loaded CrossEncoder re-ranker: '{self.model_name}'")
        except Exception as e:
            logger.warning(f"Could not load CrossEncoder model ({e}). Lexical overlap re-ranker fallback enabled.")
            self._model = None

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Re-ranks a list of retrieved chunks against the query.

        Args:
            query: User's question.
            chunks: Initial list of retrieved chunk dicts.
            top_n: Number of top chunks to return after re-ranking.

        Returns:
            Re-ranked list of chunk dicts.
        """
        if not chunks:
            return []

        if not self.enabled:
            return chunks[:top_n]

        if self._model is not None:
            try:
                pairs = [[query, chunk["text"]] for chunk in chunks]
                scores = self._model.predict(pairs)
                for i, score in enumerate(scores):
                    chunks[i]["rerank_score"] = float(score)
                chunks.sort(key=lambda c: c.get("rerank_score", 0), reverse=True)
                return chunks[:top_n]
            except Exception as e:
                logger.error(f"Cross-encoder reranking error: {e}. Falling back to lexical booster.")

        # Lexical boost fallback (exact keyword matches in text)
        query_words = set(query.lower().split())
        for chunk in chunks:
            chunk_words = set(chunk["text"].lower().split())
            overlap = len(query_words.intersection(chunk_words))
            chunk["rerank_score"] = chunk.get("score", 0.0) + (overlap * 0.05)

        chunks.sort(key=lambda c: c.get("rerank_score", 0), reverse=True)
        return chunks[:top_n]
