"""
Batch evaluation runner.
Processes test_qa.csv through the full RAG pipeline,
measures latency per stage, computes all 5 RAGAS metrics,
and saves results to eval_results.csv.
"""

import os
import time
import pandas as pd
import yaml
from typing import Optional, Dict, Any

from src.retrieval.retriever import Retriever
from src.generation.llm_service import LLMService
from src.evaluation.metrics import RAGEvaluatorMetrics
from src.utils.logger import get_logger

logger = get_logger("EvaluationRunner", "logs/eval.log")


class EvaluationRunner:
    """Orchestrates end-to-end batch evaluation of the RAG pipeline."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.test_qa_file = self.config.get("paths", {}).get("test_qa_file", "data/eval/test_qa.csv")
        self.eval_results_file = self.config.get("paths", {}).get("eval_results_file", "data/eval/eval_results.csv")

        # Initialize pipeline components
        self.retriever = Retriever(
            top_k=self.config.get("retrieval", {}).get("top_k", 5),
            use_hybrid=self.config.get("retrieval", {}).get("use_hybrid", False),
            use_reranker=self.config.get("retrieval", {}).get("use_reranker", False)
        )

        # Auto-ingest if vectorstore is empty
        if self.retriever.vectorstore.get_chunk_count() == 0:
            logger.info("Vectorstore is empty. Auto-ingesting documents from data/raw/...")
            from src.ingestion.loader import DocumentLoader
            from src.ingestion.chunker import DocumentChunker
            raw_dir = self.config.get("paths", {}).get("raw_data_dir", "data/raw")
            docs = DocumentLoader(raw_data_dir=raw_dir).load_documents()
            chunks = DocumentChunker().chunk_documents(docs)
            self.retriever.vectorstore.index_chunks(chunks)

        llm_cfg = self.config.get("llm", {})
        self.llm_service = LLMService(
            ollama_url=llm_cfg.get("ollama_url", "http://localhost:11434"),
            model_name=llm_cfg.get("model_name", "llama3.2:3b"),
            temperature=llm_cfg.get("temperature", 0.1),
            timeout=llm_cfg.get("timeout_seconds", 30)
        )

        self.evaluator = RAGEvaluatorMetrics()

    def _load_config(self, config_path: str) -> dict:
        """Loads YAML configuration file."""
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def run(self, max_samples: Optional[int] = None) -> pd.DataFrame:
        """
        Runs batch evaluation on test_qa.csv.

        Args:
            max_samples: Optional limit on number of questions to evaluate.

        Returns:
            pd.DataFrame with full evaluation results.
        """
        if not os.path.exists(self.test_qa_file):
            raise FileNotFoundError(f"Test QA file not found at: '{self.test_qa_file}'")

        df_qa = pd.read_csv(self.test_qa_file)
        if max_samples:
            df_qa = df_qa.head(max_samples)

        logger.info(f"Starting batch evaluation on {len(df_qa)} test questions...")
        results = []

        for idx, row in df_qa.iterrows():
            question = str(row["question"])
            ref_answer = str(row.get("reference_answer", ""))

            # Stage 1: Measure Retrieval
            t_retrieval_start = time.time()
            retrieved_chunks = self.retriever.retrieve(question)
            retrieval_ms = int((time.time() - t_retrieval_start) * 1000)

            # Stage 2: Measure Generation
            t_gen_start = time.time()
            gen_res = self.llm_service.generate_answer(question, retrieved_chunks)
            generation_ms = int((time.time() - t_gen_start) * 1000)

            generated_answer = gen_res["answer"]
            total_latency_ms = retrieval_ms + generation_ms

            # Stage 3: Compute RAG Metrics
            contexts = [c["text"] for c in retrieved_chunks]
            chunk_ids = [c["chunk_id"] for c in retrieved_chunks]

            scores = self.evaluator.evaluate_sample(
                question=question,
                retrieved_contexts=contexts,
                generated_answer=generated_answer,
                reference_answer=ref_answer
            )

            record = {
                "question_id": idx + 1,
                "question": question,
                "retrieved_chunk_ids": ";".join(chunk_ids),
                "generated_answer": generated_answer,
                "reference_answer": ref_answer,
                "retrieval_latency_ms": retrieval_ms,
                "generation_latency_ms": generation_ms,
                "total_latency_ms": total_latency_ms,
                "context_precision": scores["context_precision"],
                "context_recall": scores["context_recall"],
                "faithfulness": scores["faithfulness"],
                "answer_relevancy": scores["answer_relevancy"],
                "answer_correctness": scores["answer_correctness"]
            }
            results.append(record)

            logger.info(
                f"[{idx + 1}/{len(df_qa)}] Q: '{question[:35]}...' | "
                f"Faithfulness: {scores['faithfulness']} | Correctness: {scores['answer_correctness']} | "
                f"Latency: {total_latency_ms}ms"
            )

        df_results = pd.DataFrame(results)

        # Save to CSV
        os.makedirs(os.path.dirname(self.eval_results_file), exist_ok=True)
        df_results.to_csv(self.eval_results_file, index=False)
        logger.info(f"Evaluation complete! Saved report to '{self.eval_results_file}'.")

        self._print_summary(df_results)
        return df_results

    def _print_summary(self, df: pd.DataFrame):
        """Prints formatted summary scorecard of evaluation results."""
        summary = {
            "Mean Context Precision": df["context_precision"].mean(),
            "Mean Context Recall": df["context_recall"].mean(),
            "Mean Faithfulness": df["faithfulness"].mean(),
            "Mean Answer Relevancy": df["answer_relevancy"].mean(),
            "Mean Answer Correctness": df["answer_correctness"].mean(),
            "Average Total Latency (ms)": df["total_latency_ms"].mean(),
            "Average Retrieval Latency (ms)": df["retrieval_latency_ms"].mean(),
            "Average Generation Latency (ms)": df["generation_latency_ms"].mean()
        }

        print("\n" + "=" * 60)
        print("          RAG PIPELINE EVALUATION SUMMARY SCORECARD")
        print("=" * 60)
        for k, v in summary.items():
            if "Latency" in k:
                print(f"{k:<35}: {v:.1f} ms")
            else:
                print(f"{k:<35}: {v:.4f}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    runner = EvaluationRunner()
    runner.run()
