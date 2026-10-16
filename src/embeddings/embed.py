"""
Embeddings and Vector Store module.
Uses sentence-transformers/all-MiniLM-L6-v2 to produce 384-dimensional dense vectors
and ChromaDB for local persistent vector storage and similarity search.
"""

import os
import math
from typing import List, Dict, Any, Optional
from src.utils.logger import get_logger

logger = get_logger("EmbeddingService", "logs/ingestion.log")


class EmbeddingService:
    """Wrapper for sentence-transformers embedding generation."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._init_model()

    def _init_model(self):
        """Lazy initialization of SentenceTransformer model with fallback."""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Loaded embedding model: '{self.model_name}' on {self.device}")
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer ('{e}'). Fallback dummy embeddings enabled.")
            self._model = None

    def embed_text(self, text: str) -> List[float]:
        """Embeds a single query string into a 384-dim vector."""
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of text strings into a list of 384-dim vectors."""
        if not texts:
            return []

        if self._model is not None:
            try:
                embeddings = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
                return embeddings.tolist()
            except Exception as e:
                logger.error(f"Error encoding texts with model: {e}. Falling back to pseudo-embeddings.")

        # Deterministic 384-dimensional pseudo-embedding fallback for CPU/offline bootstrapping
        return [self._pseudo_embed(t, 384) for t in texts]

    def _pseudo_embed(self, text: str, dim: int = 384) -> List[float]:
        """Generates a deterministic normalized pseudo-vector based on character hashes."""
        vector = [0.0] * dim
        words = text.lower().split()
        if not words:
            return vector

        for word in words:
            h = hash(word) % dim
            vector[h] += 1.0

        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]
        return vector


class VectorStoreService:
    """Manages persistent ChromaDB vector store collection for domain documents."""

    def __init__(
        self,
        persist_dir: str = "vectorstore/chroma_db",
        collection_name: str = "domain_knowledge_base",
        embedding_service: Optional[EmbeddingService] = None
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_service = embedding_service or EmbeddingService()
        self.client = None
        self.collection = None
        self._init_chroma()

    def _init_chroma(self):
        """Initializes persistent ChromaDB client."""
        os.makedirs(self.persist_dir, exist_ok=True)
        try:
            import chromadb
            from chromadb.config import Settings

            self.client = chromadb.PersistentClient(path=self.persist_dir)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Connected to ChromaDB collection '{self.collection_name}' at '{self.persist_dir}'.")
        except Exception as e:
            logger.warning(f"ChromaDB initialization failed: {e}. In-memory fallback will be used.")
            self.client = None
            self.collection = None
            self._in_memory_store: List[Dict[str, Any]] = []

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Indexes a list of chunk dicts into ChromaDB with embeddings and metadata.

        Args:
            chunks: List of {'chunk_id': str, 'text': str, 'metadata': dict}

        Returns:
            Number of indexed chunks.
        """
        if not chunks:
            logger.warning("No chunks provided to index.")
            return 0

        texts = [c["text"] for c in chunks]
        ids = [c["chunk_id"] for c in chunks]
        metadatas = [c.get("metadata", {}) for c in chunks]

        # Generate embeddings
        embeddings = self.embedding_service.embed_documents(texts)

        if self.collection is not None:
            # Batch upsert into ChromaDB
            batch_size = 100
            total = len(chunks)
            for i in range(0, total, batch_size):
                b_ids = ids[i : i + batch_size]
                b_texts = texts[i : i + batch_size]
                b_embeds = embeddings[i : i + batch_size]
                b_meta = metadatas[i : i + batch_size]

                self.collection.upsert(
                    ids=b_ids,
                    documents=b_texts,
                    embeddings=b_embeds,
                    metadatas=b_meta
                )
            logger.info(f"Successfully indexed {total} chunks into ChromaDB collection '{self.collection_name}'.")
            return total
        else:
            # In-memory fallback
            for i, chunk in enumerate(chunks):
                self._in_memory_store.append({
                    "id": ids[i],
                    "text": texts[i],
                    "embedding": embeddings[i],
                    "metadata": metadatas[i]
                })
            logger.info(f"Indexed {len(chunks)} chunks into in-memory fallback store.")
            return len(chunks)

    def similarity_search(self, query: str, k: int = 5, filter_document: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Performs cosine similarity search for top-k matching chunks.
        Optionally filters results to a specific source_document.

        Returns:
            List of chunk dicts: [{'chunk_id': str, 'text': str, 'metadata': dict, 'score': float}]
        """
        query_vector = self.embedding_service.embed_text(query)

        if self.collection is not None and self.collection.count() > 0:
            query_kwargs = {
                "query_embeddings": [query_vector],
                "n_results": min(k, self.collection.count()),
                "include": ["documents", "metadatas", "distances"]
            }
            if filter_document:
                query_kwargs["where"] = {"source_document": filter_document}

            try:
                results = self.collection.query(**query_kwargs)
            except Exception as e:
                logger.warning(f"Filtered query failed ({e}), retrying without filter: {e}")
                results = self.collection.query(
                    query_embeddings=[query_vector],
                    n_results=min(k, self.collection.count()),
                    include=["documents", "metadatas", "distances"]
                )

            retrieved = []
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0]
            ids = results.get("ids", [[]])[0]

            for i in range(len(docs)):
                # Chroma cosine distance = 1 - cosine_similarity
                distance = dists[i] if i < len(dists) else 0.0
                similarity_score = max(0.0, 1.0 - distance)
                retrieved.append({
                    "chunk_id": ids[i] if i < len(ids) else f"chunk_{i}",
                    "text": docs[i],
                    "metadata": metas[i] if i < len(metas) else {},
                    "score": round(similarity_score, 4)
                })
            return retrieved

        elif hasattr(self, "_in_memory_store") and self._in_memory_store:
            scored = []
            candidates = self._in_memory_store
            if filter_document:
                matched = [c for c in candidates if c.get("metadata", {}).get("source_document") == filter_document]
                if matched:
                    candidates = matched

            for item in candidates:
                vec = item["embedding"]
                dot = sum(a * b for a, b in zip(query_vector, vec))
                scored.append((dot, item))
            scored.sort(key=lambda x: x[0], reverse=True)

            return [
                {
                    "chunk_id": item["id"],
                    "text": item["text"],
                    "metadata": item["metadata"],
                    "score": round(score, 4)
                }
                for score, item in scored[:k]
            ]

        return []

    def get_chunk_count(self) -> int:
        """Returns total number of chunks currently stored in vector store."""
        if self.collection is not None:
            return self.collection.count()
        if hasattr(self, "_in_memory_store"):
            return len(self._in_memory_store)
        return 0

    def get_all_source_documents(self) -> List[str]:
        """Returns list of distinct source document names indexed in vector store."""
        docs = set()
        if self.collection is not None and self.collection.count() > 0:
            try:
                res = self.collection.get(include=["metadatas"])
                for m in res.get("metadatas", []):
                    if m and "source_document" in m:
                        docs.add(m["source_document"])
            except Exception:
                pass
        if hasattr(self, "_in_memory_store"):
            for item in self._in_memory_store:
                if "source_document" in meta:
                    docs.add(meta["source_document"])
        return sorted(list(docs))

    def reset_index(self):
        """Clears the collection to allow fresh re-indexing."""
        if self.client is not None and self.collection is not None:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Reset ChromaDB collection '{self.collection_name}'.")
        if hasattr(self, "_in_memory_store"):
            self._in_memory_store = []
