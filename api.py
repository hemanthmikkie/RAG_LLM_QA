"""FastAPI REST Service for RAG-based Document Q&A System."""

import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from core.document_loader import DocumentLoaderService
from core.chunker import DocumentChunker
from core.embeddings import EmbeddingFactory
from core.vector_store import VectorStoreManager
from core.rag_chain import RAGPipeline, LLMFactory
from core.evaluation import RAGEvaluator

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="RAG Document Q&A API",
    description="REST API for Document Ingestion, Hybrid Retrieval, Grounded Question Answering, and Evaluation.",
    version="1.0.0",
)

# Enable Cross-Origin Resource Sharing (CORS) for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
def root():
    """Root endpoint welcoming users and directing to interactive documentation."""
    return {
        "service": "Enterprise RAG Document Q&A API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health",
        "status": "online",
    }


# Global State Container
class ServiceState:
    vector_manager: Optional[VectorStoreManager] = None
    rag_pipeline: Optional[RAGPipeline] = None
    indexed_documents: List[str] = []
    llm_provider: str = "gemini"   # stored at index time, used at query time

state = ServiceState()



class QueryRequest(BaseModel):
    question: str = Field(..., json_schema_extra={"example": "What is the remote work policy?"})
    retriever_type: str = Field("hybrid", json_schema_extra={"example": "hybrid"}, description="Options: 'hybrid', 'dense', 'bm25'")
    top_k: int = Field(4, ge=1, le=10, description="Number of passages to retrieve")
    llm_provider: Optional[str] = Field("gemini", json_schema_extra={"example": "gemini"}, description="Options: 'gemini', 'openai', 'groq', 'mock'")


class ChatRequest(QueryRequest):
    session_id: Optional[str] = Field("default", description="Identifier for session memory")


class CitationItem(BaseModel):
    passage_index: int
    source: str
    page: int
    chunk_id: str
    snippet: str
    full_text: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[CitationItem]
    num_passages_retrieved: int


class HealthResponse(BaseModel):
    status: str
    indexed_files: List[str]
    total_chunks: int


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check service status and index size."""
    count = state.vector_manager.count_chunks() if state.vector_manager else 0
    return {
        "status": "healthy" if state.vector_manager else "uninitialized",
        "indexed_files": state.indexed_documents,
        "total_chunks": count,
    }



@app.post("/index-samples")
def index_samples(
    embedding_provider: str = Query("huggingface", description="huggingface, gemini, openai, or mock"),
    llm_provider: str = Query("gemini", description="gemini, openai, groq, or mock"),
):
    """Index built-in sample documents to quickly get started."""
    sample_dir = os.path.join(os.path.dirname(__file__), "data", "sample_docs")
    sample_paths = [
        os.path.join(sample_dir, "company_policies.txt"),
        os.path.join(sample_dir, "rag_architecture_overview.txt"),
        os.path.join(sample_dir, "sample_ai_paper.pdf"),
    ]

    all_docs = []
    file_names = []
    for path in sample_paths:
        if os.path.exists(path):
            docs = DocumentLoaderService.load_from_file_path(path)
            all_docs.extend(docs)
            file_names.append(os.path.basename(path))

    if not all_docs:
        raise HTTPException(status_code=404, detail="Sample documents not found on disk.")

    chunker = DocumentChunker(chunk_size=600, chunk_overlap=80)
    chunks = chunker.split_documents(all_docs)

    emb = EmbeddingFactory.get_embeddings(provider=embedding_provider)
    v_mgr = VectorStoreManager(embeddings=emb)
    v_mgr.build_from_documents(chunks, save_to_disk=False)

    state.vector_manager = v_mgr
    state.rag_pipeline = None          # will be built lazily at query time
    state.llm_provider = llm_provider
    state.indexed_documents = file_names

    return {
        "message": "Sample documents successfully indexed.",
        "indexed_files": file_names,
        "total_chunks": len(chunks),
    }



@app.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    embedding_provider: str = Query("huggingface"),
    llm_provider: str = Query("gemini"),
    chunk_size: int = Query(600),
    chunk_overlap: int = Query(80),
):
    """Upload and index documents (PDF, TXT, MD, DOCX)."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    all_docs = []
    file_names = []

    for file in files:
        contents = await file.read()
        try:
            docs = DocumentLoaderService.load_from_bytes(contents, file.filename)
            all_docs.extend(docs)
            file_names.append(file.filename)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process '{file.filename}': {str(e)}")

    if not all_docs:
        raise HTTPException(status_code=400, detail="No extractable text found in uploaded files.")

    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.split_documents(all_docs)

    emb = EmbeddingFactory.get_embeddings(provider=embedding_provider)
    v_mgr = VectorStoreManager(embeddings=emb)
    v_mgr.build_from_documents(chunks, save_to_disk=False)

    state.vector_manager = v_mgr
    state.rag_pipeline = None          # built lazily at query time
    state.llm_provider = llm_provider
    state.indexed_documents = file_names

    return {
        "message": f"Successfully indexed {len(file_names)} documents into {len(chunks)} chunks.",
        "files": file_names,
        "chunks_indexed": len(chunks),
    }



@app.post("/query", response_model=QueryResponse)
def query_documents(req: QueryRequest):
    """Execute a single grounded Q&A query over the indexed knowledge base."""
    if not state.vector_manager:
        raise HTTPException(status_code=400, detail="No documents indexed. Call /upload or /index-samples first.")

    # Resolve llm_provider: request overrides state default
    provider = req.llm_provider or state.llm_provider or "mock"

    # Build or rebuild pipeline if needed
    if state.rag_pipeline is None or req.llm_provider:
        try:
            llm = LLMFactory.get_llm(provider=provider)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        retriever = state.vector_manager.get_retriever(retriever_type=req.retriever_type, top_k=req.top_k)
        state.rag_pipeline = RAGPipeline(retriever=retriever, llm=llm)
    else:
        state.rag_pipeline.retriever = state.vector_manager.get_retriever(
            retriever_type=req.retriever_type, top_k=req.top_k
        )

    result = state.rag_pipeline.query(req.question, include_history=False)
    return result


@app.post("/chat", response_model=QueryResponse)
def chat_with_documents(req: ChatRequest):
    """Multi-turn conversational Q&A endpoint retaining previous turn context."""
    if not state.vector_manager:
        raise HTTPException(status_code=400, detail="No documents indexed. Call /upload or /index-samples first.")

    provider = req.llm_provider or state.llm_provider or "mock"

    if state.rag_pipeline is None or req.llm_provider:
        try:
            llm = LLMFactory.get_llm(provider=provider)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        retriever = state.vector_manager.get_retriever(retriever_type=req.retriever_type, top_k=req.top_k)
        state.rag_pipeline = RAGPipeline(retriever=retriever, llm=llm)
    else:
        state.rag_pipeline.retriever = state.vector_manager.get_retriever(
            retriever_type=req.retriever_type, top_k=req.top_k
        )

    result = state.rag_pipeline.query(req.question, include_history=True)
    return result


@app.delete("/reset")
def reset_state():
    """Clear the vector database and conversation history."""
    state.vector_manager = None
    state.rag_pipeline = None
    state.indexed_documents = []
    return {"message": "Knowledge base and session history cleared."}
