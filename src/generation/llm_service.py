"""
LLM Service module.
Communicates with:
1. Cloud LLM (Groq API - Llama 3.3 70B / Llama 3.1 8B, fast & 100% free)
2. Local Ollama runtime (http://localhost:11434)
3. Resilient multi-chunk semantic synthesis fallback
"""

import os
import time
import json
import re
import requests
from typing import List, Dict, Any, Optional
from src.generation.prompt_builder import PromptBuilder
from src.utils.logger import get_logger

logger = get_logger("LLMService")


class LLMService:
    """Manages text generation using Groq API, local Ollama instance, or smart extractive fallback."""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model_name: str = "llama3.2:3b",
        temperature: float = 0.1,
        timeout: int = 30,
        groq_api_key: Optional[str] = None
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.model_name = model_name
        self.temperature = temperature
        self.timeout = timeout
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")
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

    def generate_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        groq_key_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates grounded answer from query and context chunks.

        Args:
            query: User's question.
            chunks: Retrieved context chunks.
            groq_key_override: Optional Groq API key passed from UI.

        Returns:
            Dict containing answer, sources, latency_ms, model_used, prompt.
        """
        prompt = PromptBuilder.build_prompt(query, chunks)
        start_time = time.time()

        # Guardrail: If no chunks or similarity score is below threshold, refuse to fabricate
        if not chunks or (chunks and chunks[0].get("score", 1.0) < 0.20):
            answer = (
                "I do not have enough information in the provided domain document to answer this question. "
                "The query appears out-of-domain or is not mentioned in the active document."
            )
            latency_ms = int((time.time() - start_time) * 1000)
            return {
                "answer": answer,
                "sources": [],
                "latency_ms": latency_ms,
                "model_used": "refusal_guardrail",
                "prompt": prompt
            }

        # 1. Option 1: Cloud Groq API (High accuracy, blazing fast, free)
        active_groq_key = groq_key_override or self.groq_api_key
        if active_groq_key and len(active_groq_key.strip()) > 10:
            try:
                headers = {
                    "Authorization": f"Bearer {active_groq_key.strip()}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {"role": "system", "content": PromptBuilder.SYSTEM_INSTRUCTION},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.temperature,
                    "max_tokens": 600
                }
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=20
                )
                if resp.status_code == 200:
                    data = resp.json()
                    answer_text = data["choices"][0]["message"]["content"].strip()
                    latency_ms = int((time.time() - start_time) * 1000)
                    sources = self._extract_sources(chunks)
                    return {
                        "answer": answer_text,
                        "sources": sources,
                        "latency_ms": latency_ms,
                        "model_used": "Groq (Llama-3.3-70B)",
                        "prompt": prompt
                    }
                else:
                    logger.warning(f"Groq API returned HTTP {resp.status_code}: {resp.text}")
            except Exception as e:
                logger.error(f"Groq API call failed: {e}")

        # 2. Option 2: Local Ollama daemon
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
                        "model_used": f"Ollama ({self.model_name})",
                        "prompt": prompt
                    }
            except Exception as e:
                logger.error(f"Ollama call failed: {e}")

        # 3. Option 3: Upgraded Multi-Chunk Grounded Synthesis Fallback
        answer_text = self._smart_synthesis_fallback(query, chunks)
        latency_ms = int((time.time() - start_time) * 1000)
        sources = self._extract_sources(chunks)

        return {
            "answer": answer_text,
            "sources": sources,
            "latency_ms": latency_ms,
            "model_used": "Offline Grounded Synthesizer",
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
                    "snippet": c.get("text", "")[:200] + "..."
                })
        return sources

    def _smart_synthesis_fallback(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Synthesizes an informative, structured answer by scoring sentences across all top chunks.
        """
        if not chunks:
            return "I do not have enough information in the provided domain documents to answer this question."

        query_tokens = set(re.findall(r"\w+", query.lower()))
        stopwords = {"what", "is", "the", "are", "of", "in", "to", "for", "and", "a", "an", "how", "can", "do", "does"}
        meaningful_query_words = query_tokens - stopwords

        scored_sentences = []
        seen_sentences = set()

        for chunk_idx, chunk in enumerate(chunks[:4]):
            raw_text = chunk.get("text", "")
            doc_name = chunk.get("source_document", "Document")
            page_num = chunk.get("page_number", 1)

            # Split by sentence or bullet point
            parts = re.split(r"(?<=[.!?])\s+|\n+", raw_text)
            for part in parts:
                cleaned_part = part.strip()
                if len(cleaned_part) < 20 or cleaned_part in seen_sentences:
                    continue

                part_words = set(re.findall(r"\w+", cleaned_part.lower()))
                overlap = len(meaningful_query_words.intersection(part_words))

                # Boost sentences containing numbers, percentages, currency, or criteria
                has_entity = bool(re.search(r"(\d+%|\bINR\b|\b\d+\b|credits?|grade|fees?|cgpa|attendance)", cleaned_part, re.I))
                score = overlap + (1.5 if has_entity and overlap > 0 else 0) - (chunk_idx * 0.2)

                if overlap > 0:
                    scored_sentences.append((score, cleaned_part, chunk_idx + 1, doc_name, page_num))
                    seen_sentences.add(cleaned_part)

        scored_sentences.sort(key=lambda x: x[0], reverse=True)

        if scored_sentences:
            top_points = scored_sentences[:4]
            lines = ["Based on verified official documentation:\n"]
            for s in top_points:
                lines.append(f"• **{s[1]}** [Source {s[2]}: {s[3]}, Page {s[4]}]")
            return "\n\n".join(lines)

        # Fallback to top chunk snippet if no direct keyword overlap
        top = chunks[0]
        snippet = top.get("text", "").strip()[:350]
        return (
            f"Based on the active domain document [{top.get('source_document')}, Page {top.get('page_number', 1)}]:\n\n"
            f"> {snippet}...\n\n"
            f"[Source 1: {top.get('source_document')}, Page {top.get('page_number', 1)}]"
        )
