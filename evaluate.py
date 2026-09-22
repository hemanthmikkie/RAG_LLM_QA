"""CLI evaluation script comparing Dense vs Hybrid retrieval and pipeline groundedness."""

import os
import sys
import json
from dotenv import load_dotenv

load_dotenv()

from core.document_loader import DocumentLoaderService
from core.chunker import DocumentChunker
from core.embeddings import EmbeddingFactory
from core.vector_store import VectorStoreManager
from core.rag_chain import RAGPipeline, LLMFactory
from core.evaluation import RAGEvaluator


def run_benchmark():
    print("=" * 60)
    print("  Enterprise RAG Benchmark: Dense vs. Hybrid Retrieval")
    print("=" * 60)

    # 1. Load sample knowledge base
    sample_dir = os.path.join(os.path.dirname(__file__), "data", "sample_docs")
    sample_paths = [
        os.path.join(sample_dir, "company_policies.txt"),
        os.path.join(sample_dir, "rag_architecture_overview.txt"),
    ]

    all_docs = []
    for path in sample_paths:
        if os.path.exists(path):
            docs = DocumentLoaderService.load_from_file_path(path)
            all_docs.extend(docs)

    print(f"Loaded {len(all_docs)} source documents.")

    # 2. Chunk documents
    chunker = DocumentChunker(chunk_size=600, chunk_overlap=80)
    chunks = chunker.split_documents(all_docs)
    print(f"Generated {len(chunks)} text chunks.")

    # 3. Initialize embeddings and vector store
    provider = os.getenv("DEFAULT_EMBEDDING_PROVIDER", "huggingface")
    try:
        emb = EmbeddingFactory.get_embeddings(provider=provider)
        print(f"Using embedding provider: '{provider}'")
    except Exception as e:
        print(f"Falling back to mock embeddings (Reason: {e})")
        emb = EmbeddingFactory.get_embeddings(provider="mock")

    v_mgr = VectorStoreManager(embeddings=emb)
    v_mgr.build_from_documents(chunks, save_to_disk=False)
    print(f"Indexed {v_mgr.count_chunks()} chunks in FAISS and BM25.")

    # 4. Run Retrieval Benchmark (Dense vs. Hybrid)
    benchmark_queries = [
        {"query": "How many days per week can employees work remotely?", "keyword": "remotely"},
        {"query": "What is the maximum reimbursement for home office equipment?", "keyword": "$1,200"},
        {"query": "What hardware refreshes are provided for engineering laptops?", "keyword": "refreshed"},
        {"query": "Why does RAG prevent LLM hallucination?", "keyword": "hallucination"},
        {"query": "What are optimal chunk sizes for technical document retrieval?", "keyword": "overlap"},
    ]

    print("\nRunning Retrieval Benchmark across 5 test queries...")
    bench_results = RAGEvaluator.compare_retrievers(v_mgr, benchmark_queries, top_k=3)

    print("\n--- Benchmark Results ---")
    print(f"Total Queries Evaluated: {bench_results['total_queries']}")
    print(f"Avg Dense Search Latency:  {bench_results['avg_dense_latency_ms']} ms")
    print(f"Avg Hybrid Search Latency: {bench_results['avg_hybrid_latency_ms']} ms")
    print("\nDetailed Query Comparisons:")
    for rec in bench_results["records"]:
        print(f"- Query: \"{rec['query']}\"")
        print(f"  Target Keyword: '{rec['target_keyword']}'")
        print(f"  Dense:  {rec['dense_keyword_hits']} hits, Latency: {rec['dense_latency_ms']} ms")
        print(f"  Hybrid: {rec['hybrid_keyword_hits']} hits, Latency: {rec['hybrid_latency_ms']} ms")

    # 5. Run Groundedness Audit
    llm_provider = os.getenv("DEFAULT_LLM_PROVIDER", "mock")
    try:
        llm = LLMFactory.get_llm(provider=llm_provider)
    except Exception:
        llm = LLMFactory.get_llm(provider="mock")

    retriever = v_mgr.get_retriever(retriever_type="hybrid", top_k=3)
    pipeline = RAGPipeline(retriever=retriever, llm=llm)

    audit_cases = [
        {"question": "What is the parental leave policy?"},
        {"question": "What is the daily meal allowance when traveling?"},
        {"question": "What is the stock option vesting schedule?"},
    ]

    print(f"\nRunning Groundedness Audit using LLM provider: '{llm_provider}'...")
    audit_results = RAGEvaluator.evaluate_pipeline(pipeline, audit_cases)

    print("\n--- Groundedness Audit Summary ---")
    print(f"Average Groundedness Score: {audit_results['average_groundedness'] * 100:.1f}%")
    print(f"Average Answer Relevance:   {audit_results['average_relevance'] * 100:.1f}%")
    print(f"Average Query Latency:      {audit_results['average_latency_sec']} s")

    print("\nDetailed Test Responses:")
    for res in audit_results["detailed_results"]:
        print(f"\nQ: {res['question']}")
        print(f"A: {res['answer']}")
        print(f"Groundedness: {res['groundedness_score']*100:.1f}% | Citations: {res['sources_count']}")

    print("\n" + "=" * 60)
    print("  Benchmark and Audit Complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()

