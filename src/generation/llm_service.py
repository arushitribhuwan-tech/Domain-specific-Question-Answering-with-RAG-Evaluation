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
        Synthesizes an informative, structured answer by extracting definitions,
        key characteristics, and examples directly from retrieved domain chunks.
        """
        if not chunks:
            return "I do not have enough information in the provided domain document to answer this question."

        stopwords = {
            "what", "is", "the", "are", "of", "in", "to", "for", "and", "a", "an",
            "how", "can", "do", "does", "explain", "describe", "define", "give",
            "tell", "me", "about", "which", "why", "who", "where", "when"
        }
        query_tokens = [w.lower() for w in re.findall(r"\w+", query) if w.lower() not in stopwords]
        if not query_tokens:
            query_tokens = [w.lower() for w in re.findall(r"\w+", query)]

        # Specific domain topic word (e.g., 'ordinal' rather than generic 'data')
        generic_words = {"data", "information", "detail", "details", "concept", "types", "system", "rule", "policy"}
        specific_terms = [t for t in query_tokens if t not in generic_words]
        target_term = specific_terms[0] if specific_terms else (query_tokens[0] if query_tokens else "")

        definitions = []
        characteristics = []
        examples = []
        notes = []
        supporting_sentences = []
        seen = set()

        for chunk_idx, chunk in enumerate(chunks[:5]):
            raw_text = chunk.get("text", "")
            raw_lines = [l.strip() for l in raw_text.split("\n") if l.strip()]

            current_topic_active = bool(target_term and target_term in raw_text.lower())
            in_example_block = False

            for i, line in enumerate(raw_lines):
                norm_line = line.lower().strip()

                # Filter pure page markers, chapter numbers, or section headers
                if re.match(r"^(page\s+\d+|\d+(\.\d+)*\s+[A-Za-z\s]+$|characteristics.*:$)", norm_line):
                    if target_term and target_term in norm_line:
                        current_topic_active = True
                    elif any(other in norm_line for other in ["nominal data", "binary data", "continuous", "numerical", "graphical representation"]):
                        if target_term not in norm_line:
                            current_topic_active = False
                    in_example_block = False
                    continue

                # Check if this line is a "Definition" header
                if norm_line == "definition":
                    if i + 1 < len(raw_lines):
                        def_candidate = raw_lines[i + 1].strip()
                        if i + 2 < len(raw_lines) and not def_candidate.endswith((".", ":", "!", "?")):
                            def_candidate += " " + raw_lines[i + 2].strip()
                        clean_def = re.sub(r"^(definition\s*:?|\d+(\.\d+)*\s*)", "", def_candidate, flags=re.I).strip()
                        if target_term in def_candidate.lower() and clean_def.lower() not in seen:
                            definitions.append(clean_def)
                            seen.add(clean_def.lower())
                            current_topic_active = True
                    continue

                # Check for "Examples:" header
                if norm_line.startswith("example"):
                    in_example_block = True
                    continue

                if norm_line in seen or len(norm_line) < 6:
                    continue

                clean_text = re.sub(r"^[•\-\*\u2022\uFFFD\d\.\)]\s*", "", line).strip()
                clean_text = re.sub(r"^(definition\s*:?|important point\s*:?|\d+(\.\d+)*\s*)", "", clean_text, flags=re.I).strip()

                is_bullet = bool(re.match(r"^[•\-\*\u2022\uFFFD]", line))

                if is_bullet:
                    in_example_block = False

                # Extract definitions matching inline sentence patterns
                if target_term and target_term in norm_line and not definitions:
                    if re.search(r"\b(called|defined as|refers to|means|is defined|possess|consists of)\b", norm_line):
                        def_candidate = clean_text
                        if i + 1 < len(raw_lines) and not def_candidate.endswith((".", "!", "?")):
                            def_candidate += " " + raw_lines[i + 1].strip()
                        definitions.append(def_candidate)
                        seen.add(norm_line)
                        current_topic_active = True
                        continue

                # Collect examples
                if in_example_block and current_topic_active and len(examples) < 5:
                    if not re.match(r"^(variable|basis|field|categories|meaning|ranking)", norm_line) and len(clean_text) > 4:
                        examples.append(clean_text)
                        seen.add(norm_line)
                # Collect bullet points / characteristics
                elif is_bullet and current_topic_active and len(characteristics) < 5:
                    if "no ranking" not in norm_line and clean_text.lower() not in seen:
                        characteristics.append(clean_text)
                        seen.add(clean_text.lower())
                # Collect important notes / distinctions
                elif "important point" in norm_line or (i > 0 and "important point" in raw_lines[i - 1].lower()):
                    if current_topic_active and len(notes) < 2 and norm_line != "important point":
                        note_candidate = clean_text
                        if i + 1 < len(raw_lines) and not note_candidate.endswith((".", "!", "?")):
                            note_candidate += " " + raw_lines[i + 1].strip()
                        notes.append(note_candidate)
                        seen.add(norm_line)
                # General high-overlap sentences as backup
                elif target_term in norm_line and len(supporting_sentences) < 4:
                    if len(clean_text) > 25 and not re.match(r"^\d+\.\d+", line):
                        supporting_sentences.append(clean_text)
                        seen.add(norm_line)

        sections = []
        if definitions:
            def_text = "\n\n".join(definitions)
            sections.append(f"### 📌 Definition\n{def_text}")
        elif supporting_sentences:
            sections.append(f"### 📌 Overview\n{supporting_sentences[0]}")

        if characteristics:
            char_text = "\n".join(f"• {c}" for c in characteristics)
            sections.append(f"### 🔑 Key Characteristics\n{char_text}")

        if examples:
            clean_ex = [e for e in examples if len(e.strip()) > 3]
            if clean_ex:
                ex_text = "\n".join(f"• {e}" for e in clean_ex)
                sections.append(f"### 💡 Examples from Document\n{ex_text}")

        if notes:
            clean_notes = [n for n in notes if n.lower() != "important point" and len(n.strip()) > 10]
            if clean_notes:
                sections.append(f"### ℹ️ Key Note\n" + "\n\n".join(clean_notes))

        if not sections and supporting_sentences:
            sections.append("### 📌 Verified Facts\n" + "\n".join(f"• {p}" for p in supporting_sentences[:3]))

        if not sections:
            top = chunks[0]
            snippet = top.get("text", "").strip()[:400]
            return f"Based on the active domain document:\n\n> {snippet}..."

        return "\n\n".join(sections)
