"""Integration tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_health_uninitialized():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data


def test_index_samples_and_query_endpoints():
    # 1. Index sample docs using mock provider for zero-cost offline test
    index_res = client.post("/index-samples?embedding_provider=mock&llm_provider=mock")
    assert index_res.status_code == 200
    index_data = index_res.json()
    assert index_data["total_chunks"] > 0
    assert len(index_data["indexed_files"]) == 3

    # 2. Query endpoint
    query_res = client.post(
        "/query",
        json={
            "question": "What is the policy on remote work?",
            "retriever_type": "hybrid",
            "top_k": 2,
            "llm_provider": "mock",
        },
    )
    assert query_res.status_code == 200
    q_data = query_res.json()
    assert "answer" in q_data
    assert "sources" in q_data
    assert len(q_data["sources"]) == 2

    # 3. Chat endpoint
    chat_res = client.post(
        "/chat",
        json={
            "question": "Can I have an engineering laptop refreshed?",
            "retriever_type": "dense",
            "top_k": 2,
            "llm_provider": "mock",
        },
    )
    assert chat_res.status_code == 200
    c_data = chat_res.json()
    assert "answer" in c_data

    # 4. Health status check
    health_res = client.get("/health")
    assert health_res.status_code == 200
    assert health_res.json()["status"] == "healthy"

    # 5. Reset endpoint
    reset_res = client.delete("/reset")
    assert reset_res.status_code == 200

