"""
LLM Service module.
Communicates with local Ollama runtime (http://localhost:11434)
and provides resilient fallback handling with latency profiling.
"""

import time
import json
import requests
from typing import List, Dict, Any, Optional
from src.generation.prompt_builder import PromptBuilder
from src.utils.logger import get_logger

logger = get_logger("LLMService")


class LLMService:
    """Manages text generation using local Ollama instance or fallback."""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model_name: str = "llama3.2:3b",
        temperature: float = 0.1,
        timeout: int = 30
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.model_name = model_name
        self.temperature = temperature
        self.timeout = timeout
        self._last_ollama_check_time = 0
        self._ollama_online_cache = None

    def check_ollama_status(self) -> bool:
        """Checks if local Ollama daemon is reachable with 1s timeout."""
        current_time = time.time()
        if self._ollama_online_cache is not None and (current_time - self._last_ollama_check_time) < 15:
            return self._ollama_online_cache

        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=1.0)
            self._ollama_online_cache = (resp.status_code == 200)
        except Exception:
            self._ollama_online_cache = False

        self._last_ollama_check_time = current_time
        return self._ollama_online_cache

    def generate_answer(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates grounded answer from query and context chunks.

        Args:
            query: User's question.
            chunks: Retrieved context chunks.

        Returns:
            Dict containing:
            {
                'answer': str,
                'sources': list,
                'latency_ms': int,
                'model_used': str,
                'prompt': str
            }
        """
        prompt = PromptBuilder.build_prompt(query, chunks)
        start_time = time.time()

        # If no chunks were retrieved or highest similarity is too low, reject early
        if not chunks or (chunks and chunks[0].get("score", 1.0) < 0.20):
            answer = (
                "I do not have enough information in the provided domain documents to answer this question. "
                "The query appears out-of-domain or not present in the indexed regulations."
            )
            latency_ms = int((time.time() - start_time) * 1000)
            return {
                "answer": answer,
                "sources": [],
                "latency_ms": latency_ms,
                "model_used": "refusal_filter",
                "prompt": prompt
            }

        # Attempt Ollama inference if server is running
        if self.check_ollama_status():
            try:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature
                    }
                }

                resp = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                    timeout=self.timeout
                )

                if resp.status_code == 200:
                    data = resp.json()
                    answer_text = data.get("response", "").strip()
                    latency_ms = int((time.time() - start_time) * 1000)
                    sources = self._extract_sources(chunks)

                    return {
                        "answer": answer_text,
                        "sources": sources,
                        "latency_ms": latency_ms,
                        "model_used": self.model_name,
                        "prompt": prompt
                    }
                else:
                    logger.warning(f"Ollama returned HTTP {resp.status_code}: {resp.text}. Using fallback.")

            except requests.exceptions.ConnectionError:
                logger.warning("Ollama service not detected at http://localhost:11434. Operating in extractive fallback mode.")
            except Exception as e:
                logger.error(f"Error calling LLM: {e}. Operating in extractive fallback mode.")

        # Resilient offline extractive fallback:
        # Synthesizes an answer directly from the top retrieved chunks so the app runs without crashing
        answer_text = self._extractive_fallback(query, chunks)
        latency_ms = int((time.time() - start_time) * 1000)
        sources = self._extract_sources(chunks)

        return {
            "answer": answer_text,
            "sources": sources,
            "latency_ms": latency_ms,
            "model_used": f"{self.model_name} (Local-Offline-Fallback)",
            "prompt": prompt
        }

    def _extract_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Formats unique source citations from chunks."""
        seen = set()
        sources = []
        for c in chunks:
            key = (c.get("source_document"), c.get("page_number"))
            if key not in seen and c.get("source_document"):
                seen.add(key)
                sources.append({
                    "source_document": c.get("source_document"),
                    "page_number": c.get("page_number", 1),
                    "chunk_id": c.get("chunk_id"),
                    "score": c.get("score", 0.0),
                    "snippet": c.get("text", "")[:180] + "..."
                })
        return sources

    def _extractive_fallback(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Grounded fallback extractor when local Ollama daemon is offline."""
        if not chunks:
            return "I do not have enough information in the provided domain documents to answer this question."

        top_chunk = chunks[0]
        text = top_chunk.get("text", "")

        # Find the most relevant sentence from the top chunk
        sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
        query_words = set(query.lower().split())

        best_sentence = ""
        best_overlap = -1

        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(query_words.intersection(sentence_words))
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = sentence

        if best_sentence and best_overlap > 0:
            return (
                f"Based on [Source 1: {top_chunk.get('source_document', 'Document')}, "
                f"Page {top_chunk.get('page_number', 1)}]:\n\n"
                f"{best_sentence}.\n\n"
                f"*(Note: Generated via extractive grounded fallback as Ollama was offline)*"
            )

        return (
            f"Based on the domain documentation [Source 1]:\n\n{text[:250]}...\n\n"
            f"*(Note: Generated via extractive grounded fallback as Ollama was offline)*"
        )
