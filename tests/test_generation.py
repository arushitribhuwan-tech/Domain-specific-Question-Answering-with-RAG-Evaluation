"""
Unit tests for Generation & Evaluation Modules (prompt builder, refusal filter, metrics).
"""

try:
    import pytest
except ImportError:
    pytest = None
from src.generation.prompt_builder import PromptBuilder
from src.generation.llm_service import LLMService
from src.evaluation.metrics import RAGEvaluatorMetrics


def test_prompt_builder_structure():
    query = "What is the passing criteria?"
    chunks = [
        {
            "source_document": "eval_rules.pdf",
            "page_number": 5,
            "text": "Passing requires 40 percent in End Semester Examinations."
        }
    ]

    prompt = PromptBuilder.build_prompt(query, chunks)
    assert "=== VERIFIED CONTEXT ===" in prompt
    assert "[Source 1: eval_rules.pdf, Page 5]" in prompt
    assert "=== USER QUESTION ===" in prompt
    assert query in prompt
    assert "CRITICAL RULES" in prompt


def test_out_of_domain_refusal():
    llm = LLMService()
    # Empty retrieved chunks indicates out-of-domain / unretrievable topic
    res = llm.generate_answer("How do I bake a cake?", chunks=[])
    ans_lower = res["answer"].lower()
    assert "not enough information" in ans_lower or "do not have enough information" in ans_lower
    assert res["sources"] == []


def test_evaluation_metrics_computation():
    evaluator = RAGEvaluatorMetrics()
    question = "What is the minimum attendance required?"
    context = ["Every student must maintain at least 75% attendance to sit for exams."]
    gen_answer = "Students must maintain at least 75% attendance for exams."
    ref_answer = "A minimum attendance of 75% is required for exam eligibility."

    scores = evaluator.evaluate_sample(
        question=question,
        retrieved_contexts=context,
        generated_answer=gen_answer,
        reference_answer=ref_answer
    )

    assert 0.0 <= scores["context_precision"] <= 1.0
    assert 0.0 <= scores["context_recall"] <= 1.0
    assert 0.0 <= scores["faithfulness"] <= 1.0
    assert 0.0 <= scores["answer_relevancy"] <= 1.0
    assert 0.0 <= scores["answer_correctness"] <= 1.0
    assert scores["faithfulness"] > 0.5
    assert scores["answer_relevancy"] > 0.5
