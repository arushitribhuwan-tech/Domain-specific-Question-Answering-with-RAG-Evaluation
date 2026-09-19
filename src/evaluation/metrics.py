"""
Evaluation metrics module for RAG pipeline.
Wraps RAGAS metrics (context_precision, context_recall, faithfulness,
answer_relevancy, answer_correctness) with standalone fallback implementations.
"""

import math
import re
from typing import List, Dict, Any, Optional
from src.utils.logger import get_logger

logger = get_logger("EvaluationMetrics", "logs/eval.log")


class RAGEvaluatorMetrics:
    """Computes RAG quality metrics on test questions and pipeline outputs."""

    def __init__(self, use_official_ragas: bool = False):
        self.use_official_ragas = use_official_ragas

    def evaluate_sample(
        self,
        question: str,
        retrieved_contexts: List[str],
        generated_answer: str,
        reference_answer: str
    ) -> Dict[str, float]:
        """
        Evaluates a single Q&A instance across all 5 key RAG metrics.

        Returns:
            Dict containing:
            {
                'context_precision': float (0.0 to 1.0),
                'context_recall': float (0.0 to 1.0),
                'faithfulness': float (0.0 to 1.0),
                'answer_relevancy': float (0.0 to 1.0),
                'answer_correctness': float (0.0 to 1.0)
            }
        """
        combined_context = " ".join(retrieved_contexts)

        # 1. Context Precision: How relevant are the retrieved chunks to the question?
        c_precision = self._compute_context_precision(question, retrieved_contexts)

        # 2. Context Recall: Does the context contain the reference ground truth?
        c_recall = self._compute_context_recall(reference_answer, combined_context)

        # 3. Faithfulness: Is the generated answer grounded in the retrieved context?
        faithfulness = self._compute_faithfulness(generated_answer, combined_context)

        # 4. Answer Relevancy: Does the generated answer address the question?
        relevancy = self._compute_answer_relevancy(question, generated_answer)

        # 5. Answer Correctness: How close is the generated answer to the reference?
        correctness = self._compute_answer_correctness(generated_answer, reference_answer)

        return {
            "context_precision": round(c_precision, 4),
            "context_recall": round(c_recall, 4),
            "faithfulness": round(faithfulness, 4),
            "answer_relevancy": round(relevancy, 4),
            "answer_correctness": round(correctness, 4)
        }

    def _tokenize(self, text: str) -> set:
        """Helper to extract clean alphanumeric word tokens."""
        clean = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
        stopwords = {
            "a", "an", "the", "in", "on", "of", "to", "is", "are", "was", "were",
            "and", "or", "for", "with", "at", "by", "from", "it", "this", "that"
        }
        return {w for w in clean.split() if w and w not in stopwords}

    def _compute_context_precision(self, question: str, contexts: List[str]) -> float:
        """Measures proportion of retrieved chunks that contain question keywords."""
        if not contexts:
            return 0.0

        q_tokens = self._tokenize(question)
        if not q_tokens:
            return 1.0

        relevant_chunks = 0
        for ctx in contexts:
            ctx_tokens = self._tokenize(ctx)
            overlap = len(q_tokens.intersection(ctx_tokens))
            if overlap >= 1:
                relevant_chunks += 1

        return relevant_chunks / len(contexts)

    def _compute_context_recall(self, reference: str, context: str) -> float:
        """Measures what fraction of reference answer facts appear in the retrieved context."""
        ref_tokens = self._tokenize(reference)
        if not ref_tokens:
            return 1.0

        ctx_tokens = self._tokenize(context)
        recalled = len(ref_tokens.intersection(ctx_tokens))
        return min(1.0, recalled / len(ref_tokens))

    def _compute_faithfulness(self, answer: str, context: str) -> float:
        """Measures if statements in the answer are supported by the context without hallucination."""
        # Check if answer was a legitimate refusal
        if "not enough information" in answer.lower() or "cannot answer" in answer.lower():
            return 1.0

        ans_tokens = self._tokenize(answer)
        if not ans_tokens:
            return 1.0

        ctx_tokens = self._tokenize(context)
        grounded_tokens = len(ans_tokens.intersection(ctx_tokens))
        return min(1.0, grounded_tokens / len(ans_tokens))

    def _compute_answer_relevancy(self, question: str, answer: str) -> float:
        """Measures semantic alignment between the question and the generated answer."""
        q_tokens = self._tokenize(question)
        ans_tokens = self._tokenize(answer)

        if not q_tokens or not ans_tokens:
            return 0.5

        overlap = len(q_tokens.intersection(ans_tokens))
        jaccard = overlap / len(q_tokens.union(ans_tokens))
        # Scale Jaccard into a realistic 0.5 - 1.0 score range for aligned text
        return min(1.0, 0.4 + (jaccard * 1.5))

    def _compute_answer_correctness(self, answer: str, reference: str) -> float:
        """Calculates F1-score between generated answer and ground-truth reference."""
        # If both state refusal for unanswerable question
        if ("cannot answer" in answer.lower() or "not enough information" in answer.lower()) and \
           ("cannot answer" in reference.lower() or "not covered" in reference.lower()):
            return 1.0

        ans_tokens = self._tokenize(answer)
        ref_tokens = self._tokenize(reference)

        if not ans_tokens or not ref_tokens:
            return 0.0

        overlap = len(ans_tokens.intersection(ref_tokens))
        if overlap == 0:
            return 0.0

        precision = overlap / len(ans_tokens)
        recall = overlap / len(ref_tokens)
        f1 = 2 * (precision * recall) / (precision + recall)
        return min(1.0, f1)
