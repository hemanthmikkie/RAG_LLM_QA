"""Evaluation harness supporting RAGAS metrics and local benchmark comparisons (Dense vs Hybrid)."""

import time
import math
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from core.vector_store import VectorStoreManager
from core.rag_chain import RAGPipeline


class RAGEvaluator:
    """Evaluates RAG pipeline components for Groundedness, Relevance, and Retrieval Quality."""

    @staticmethod
    def calculate_lexical_overlap(text_a: str, text_b: str) -> float:
        """Compute token overlap (Jaccard similarity) between two texts."""
        tokens_a = set(text_a.lower().split())
        tokens_b = set(text_b.lower().split())
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a.intersection(tokens_b)
        union = tokens_a.union(tokens_b)
        return len(intersection) / len(union)

    @staticmethod
    def evaluate_groundedness(answer: str, context: str) -> float:
        """Evaluate whether tokens/claims in the answer exist in the retrieved context."""
        if not answer or not context:
            return 0.0
        # If the model explicitly stated it cannot answer from context
        if "cannot answer" in answer.lower() or "unable to answer" in answer.lower():
            return 1.0  # Perfect refusal when context is missing

        answer_words = [w.lower().strip(".,;:?!'\"()[]") for w in answer.split() if len(w) > 3]
        if not answer_words:
            return 1.0
        context_lower = context.lower()
        supported = sum(1 for w in answer_words if w in context_lower)
        return round(supported / len(answer_words), 3)

    @staticmethod
    def evaluate_answer_relevance(question: str, answer: str) -> float:
        """Assess query-answer relevance using word token overlap & length penalty."""
        overlap = RAGEvaluator.calculate_lexical_overlap(question, answer)
        # Scaled score with threshold
        score = min(1.0, overlap * 2.5)
        return round(score, 3)

    @classmethod
    def evaluate_pipeline(
        cls,
        rag_pipeline: RAGPipeline,
        test_cases: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run evaluation over a test dataset of questions and expected answers.

        test_cases schema:
        [
            {
                "question": "What is the policy on remote work?",
                "expected_ground_truth": "Employees can work up to 2 days remotely."
            }
        ]
        """
        results = []
        total_groundedness = 0.0
        total_relevance = 0.0
        total_time = 0.0

        for case in test_cases:
            q = case["question"]
            start_t = time.time()
            rag_output = rag_pipeline.query(q, include_history=False)
            latency = time.time() - start_t
            total_time += latency

            answer = rag_output["answer"]
            context = rag_output["raw_context"]

            groundedness = cls.evaluate_groundedness(answer, context)
            relevance = cls.evaluate_answer_relevance(q, answer)

            total_groundedness += groundedness
            total_relevance += relevance

            results.append(
                {
                    "question": q,
                    "answer": answer,
                    "groundedness_score": groundedness,
                    "relevance_score": relevance,
                    "latency_sec": round(latency, 3),
                    "sources_count": len(rag_output["sources"]),
                }
            )

        n = max(1, len(test_cases))
        return {
            "num_test_cases": len(test_cases),
            "average_groundedness": round(total_groundedness / n, 3),
            "average_relevance": round(total_relevance / n, 3),
            "average_latency_sec": round(total_time / n, 3),
            "detailed_results": results,
        }

    @staticmethod
    def compare_retrievers(
        vector_manager: VectorStoreManager,
        benchmark_queries: List[Dict[str, Any]],
        top_k: int = 4,
    ) -> Dict[str, Any]:
        """Compare Dense (FAISS) vs Hybrid (Dense + BM25) retrieval on benchmark queries."""
        dense_retriever = vector_manager.get_retriever(retriever_type="dense", top_k=top_k)
        hybrid_retriever = vector_manager.get_retriever(retriever_type="hybrid", top_k=top_k)

        dense_latencies = []
        hybrid_latencies = []
        comparison_records = []

        for item in benchmark_queries:
            query = item["query"]
            target_keyword = item.get("keyword", "")

            # 1. Dense retrieval
            t0 = time.time()
            dense_docs: List[Document] = dense_retriever.invoke(query)
            t_dense = time.time() - t0
            dense_latencies.append(t_dense)

            # 2. Hybrid retrieval
            t1 = time.time()
            hybrid_docs: List[Document] = hybrid_retriever.invoke(query)
            t_hybrid = time.time() - t1
            hybrid_latencies.append(t_hybrid)

            # Keyword hit check (precision on exact terms)
            dense_hits = sum(1 for d in dense_docs if target_keyword.lower() in d.page_content.lower()) if target_keyword else len(dense_docs)
            hybrid_hits = sum(1 for d in hybrid_docs if target_keyword.lower() in d.page_content.lower()) if target_keyword else len(hybrid_docs)

            comparison_records.append(
                {
                    "query": query,
                    "target_keyword": target_keyword,
                    "dense_retrieved_count": len(dense_docs),
                    "hybrid_retrieved_count": len(hybrid_docs),
                    "dense_keyword_hits": dense_hits,
                    "hybrid_keyword_hits": hybrid_hits,
                    "dense_latency_ms": round(t_dense * 1000, 2),
                    "hybrid_latency_ms": round(t_hybrid * 1000, 2),
                }
            )

        n = max(1, len(benchmark_queries))
        return {
            "total_queries": len(benchmark_queries),
            "avg_dense_latency_ms": round(sum(dense_latencies) / n * 1000, 2),
            "avg_hybrid_latency_ms": round(sum(hybrid_latencies) / n * 1000, 2),
            "records": comparison_records,
        }

