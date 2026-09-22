"""Unit and integration tests for RAG pipeline components."""

import os
import pytest
from langchain_core.documents import Document
from core.document_loader import clean_text, DocumentLoaderService
from core.chunker import DocumentChunker
from core.embeddings import EmbeddingFactory
from core.vector_store import VectorStoreManager
from core.rag_chain import RAGPipeline, LLMFactory, format_context_with_citations, extract_citations
from core.evaluation import RAGEvaluator


def test_clean_text():
    raw = "   Line 1\r\n\r\n\r\nLine 2   with   spaces\t\t\n\n\n\nLine 3   "
    cleaned = clean_text(raw)
    assert "Line 1" in cleaned
    assert "Line 2 with spaces" in cleaned
    assert "\r" not in cleaned
    assert "   " not in cleaned


def test_document_loader_txt():
    sample_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs", "company_policies.txt")
    docs = DocumentLoaderService.load_from_file_path(sample_path)
    assert len(docs) > 0
    assert "company_policies.txt" in docs[0].metadata["source"]
    assert "Acme Corporation" in docs[0].page_content


def test_chunker_metadata():
    docs = [
        Document(
            page_content="Artificial intelligence and retrieval-augmented generation allow models to ground their answers in custom documents. Overlapping chunks ensure that split sentences remain coherent across boundaries.",
            metadata={"source": "test.txt", "page": 1},
        )
    ]
    chunker = DocumentChunker(chunk_size=80, chunk_overlap=20)
    chunks = chunker.split_documents(docs)

    assert len(chunks) >= 2
    for idx, c in enumerate(chunks):
        assert c.metadata["chunk_index"] == idx
        assert "chunk_id" in c.metadata
        assert "test.txt" in c.metadata["chunk_id"]
        assert c.metadata["char_count"] == len(c.page_content)


def test_vector_store_dense_and_hybrid():
    # Use FakeEmbeddings for fast, offline testing
    emb = EmbeddingFactory.get_embeddings(provider="mock")
    v_mgr = VectorStoreManager(embeddings=emb)

    docs = [
        Document(page_content="Acme allows employees to work remotely up to 3 days per week.", metadata={"source": "policy.txt", "page": 1}),
        Document(page_content="Engineering laptops are refreshed every three years.", metadata={"source": "tech.txt", "page": 1}),
        Document(page_content="Full-time employees receive twenty-two days of paid time off.", metadata={"source": "leaves.txt", "page": 1}),
    ]

    v_mgr.build_from_documents(docs, save_to_disk=False)
    assert v_mgr.count_chunks() == 3

    # Test Dense Retriever
    dense_retriever = v_mgr.get_retriever(retriever_type="dense", top_k=2)
    dense_results = dense_retriever.invoke("remote work")
    assert len(dense_results) == 2

    # Test Hybrid Retriever
    hybrid_retriever = v_mgr.get_retriever(retriever_type="hybrid", top_k=2)
    hybrid_results = hybrid_retriever.invoke("laptops refreshed")
    assert len(hybrid_results) > 0


def test_rag_pipeline_and_citations():
    emb = EmbeddingFactory.get_embeddings(provider="mock")
    v_mgr = VectorStoreManager(embeddings=emb)

    docs = [
        Document(page_content="Parental leave covers 16 weeks of fully paid leave.", metadata={"source": "hr.txt", "page": 2, "chunk_id": "hr::p2::c0"}),
    ]
    v_mgr.build_from_documents(docs, save_to_disk=False)

    retriever = v_mgr.get_retriever(retriever_type="dense", top_k=1)
    mock_llm = LLMFactory.get_llm(provider="mock")

    pipeline = RAGPipeline(retriever=retriever, llm=mock_llm)

    res = pipeline.query("How long is parental leave?", include_history=True)
    assert "answer" in res
    assert "sources" in res
    assert len(res["sources"]) == 1
    assert res["sources"][0]["source"] == "hr.txt"
    assert res["sources"][0]["page"] == 2

    # Verify conversation memory state
    assert len(pipeline.chat_history) == 2  # Human + AI message
    pipeline.clear_history()
    assert len(pipeline.chat_history) == 0


def test_evaluation_metrics():
    context = "Acme offers 16 weeks of fully paid parental leave."
    grounded_answer = "Employees receive 16 weeks of paid parental leave."
    hallucinated_answer = "Employees receive 50 weeks of free vacations in Mars."

    score_grounded = RAGEvaluator.evaluate_groundedness(grounded_answer, context)
    score_hallucinated = RAGEvaluator.evaluate_groundedness(hallucinated_answer, context)

    assert score_grounded > score_hallucinated
    assert score_grounded >= 0.6

