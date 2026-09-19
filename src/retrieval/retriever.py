"""
Retriever module.
Executes vector similarity search against ChromaDB, supports optional BM25 hybrid search,
and passes candidates through the optional re-ranker.
"""

from typing import List, Dict, Any, Optional
from src.embeddings.embed import VectorStoreService
from src.retrieval.reranker import ReRanker
from src.utils.logger import get_logger

logger = get_logger("Retriever")


class Retriever:
    """Manages search queries across the vector store and optional hybrid indices."""

    def __init__(
        self,
        vectorstore_service: Optional[VectorStoreService] = None,
        top_k: int = 5,
        use_hybrid: bool = False,
        use_reranker: bool = False
    ):
        self.vectorstore = vectorstore_service or VectorStoreService()
        self.top_k = top_k
        self.use_hybrid = use_hybrid
        self.reranker = ReRanker(enabled=use_reranker)
        self.bm25 = None
        self.corpus_chunks: List[Dict[str, Any]] = []

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieves top-k relevant chunks for a given user query.

        Args:
            query: Question or search string.
            k: Optional override for top-k results.

        Returns:
            List of chunk dicts:
            [{
                'chunk_id': str,
                'source_document': str,
                'page_number': int,
                'text': str,
                'score': float
            }]
        """
        effective_k = k or self.top_k

        if not query or not query.strip():
            return []

        # 1. Dense vector similarity search
        vector_results = self.vectorstore.similarity_search(query, k=effective_k * 2)

        if not vector_results:
            logger.warning(f"No vector chunks found for query: '{query}'")
            return []

        candidates = vector_results

        # 2. Optional BM25 Hybrid Retrieval
        if self.use_hybrid and self.bm25 is not None:
            candidates = self._combine_bm25_vector(query, vector_results)

        # 3. Optional Re-ranking
        if self.reranker.enabled:
            candidates = self.reranker.rerank(query, candidates, top_n=effective_k)
        else:
            candidates = candidates[:effective_k]

        # Format chunk representation strictly as defined in TRD
        formatted_chunks = []
        for c in candidates:
            metadata = c.get("metadata", {})
            formatted_chunks.append({
                "chunk_id": c.get("chunk_id", metadata.get("chunk_id", "unknown")),
                "source_document": metadata.get("source_document", "unknown"),
                "page_number": int(metadata.get("page_number", 1)),
                "text": c.get("text", ""),
                "score": float(c.get("score", 0.0))
            })

        logger.info(f"Retrieved {len(formatted_chunks)} chunks for query '{query[:40]}...'")
        return formatted_chunks

    def _combine_bm25_vector(self, query: str, vector_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Blends BM25 keyword matching scores with dense vector scores via Reciprocal Rank Fusion (RRF)."""
        try:
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25.get_scores(tokenized_query)

            # Map chunk_id to rrf score
            rrf_scores: Dict[str, float] = {}
            k_rrf = 60

            # Rank from vector results
            for rank, item in enumerate(vector_results):
                cid = item["chunk_id"]
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k_rrf + rank + 1))

            # Combine
            for item in vector_results:
                cid = item["chunk_id"]
                item["score"] = rrf_scores.get(cid, item.get("score", 0.0))

            vector_results.sort(key=lambda x: x["score"], reverse=True)
            return vector_results
        except Exception as e:
            logger.warning(f"BM25 blending failed: {e}. Using pure vector results.")
            return vector_results
