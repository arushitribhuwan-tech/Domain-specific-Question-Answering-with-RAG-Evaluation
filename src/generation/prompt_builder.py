"""
Prompt builder module.
Constructs strict, grounded prompts restricting the LLM to provided context
and formatting citations.
"""

from typing import List, Dict, Any


class PromptBuilder:
    """Constructs grounded prompts for domain-specific RAG generation."""

    SYSTEM_INSTRUCTION = (
        "You are an authoritative, accurate domain assistant. Your job is to answer the user's question "
        "STRICTLY using ONLY the provided verified context chunks below.\n\n"
        "CRITICAL RULES:\n"
        "1. Do NOT assume, extrapolate, or use outside training knowledge.\n"
        "2. If the provided context does NOT contain enough facts to answer the question, you MUST respond exactly with:\n"
        "   'I do not have enough information in the provided domain documents to answer this question.'\n"
        "3. Explicitly cite which Source number(s) (e.g., [Source 1], [Source 2]) your answer was derived from.\n"
        "4. Keep the answer direct, factual, and professional."
    )

    @classmethod
    def build_prompt(cls, query: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Builds complete text prompt formatted with numbered context chunks.

        Args:
            query: User's question string.
            chunks: List of retrieved chunk dictionaries.

        Returns:
            Formatted prompt string ready for LLM inference.
        """
        context_blocks = []

        if not chunks:
            formatted_context = "No relevant context found in database."
        else:
            for i, chunk in enumerate(chunks):
                doc_name = chunk.get("source_document", "Unknown Document")
                page_num = chunk.get("page_number", 1)
                chunk_text = chunk.get("text", "").strip()

                context_blocks.append(
                    f"--- [Source {i + 1}: {doc_name}, Page {page_num}] ---\n{chunk_text}"
                )
            formatted_context = "\n\n".join(context_blocks)

        prompt = (
            f"{cls.SYSTEM_INSTRUCTION}\n\n"
            f"=== VERIFIED CONTEXT ===\n"
            f"{formatted_context}\n\n"
            f"=== USER QUESTION ===\n"
            f"{query}\n\n"
            f"=== GROUNDED ANSWER (Include citations like [Source X]) ==="
        )

        return prompt
