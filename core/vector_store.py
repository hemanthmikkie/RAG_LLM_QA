"""Vector store management with FAISS, BM25 keyword search, and hybrid ensemble retrieval."""

import os
from typing import List, Optional, Tuple, Dict, Any
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    try:
        from langchain_classic.retrievers.ensemble import EnsembleRetriever
    except ImportError:
        try:
            from langchain_community.retrievers import EnsembleRetriever
        except ImportError:
            # Custom Rank-Fused EnsembleRetriever fallback
            class EnsembleRetriever(BaseRetriever):
                retrievers: List[BaseRetriever]
                weights: List[float] = [0.5, 0.5]

                def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
                    doc_scores = {}
                    for retriever, weight in zip(self.retrievers, self.weights):
                        docs = retriever.invoke(query)
                        for rank, doc in enumerate(docs):
                            key = doc.page_content
                            rrf_score = weight * (1.0 / (60 + rank))
                            if key in doc_scores:
                                doc_scores[key]["score"] += rrf_score
                            else:
                                doc_scores[key] = {"doc": doc, "score": rrf_score}
                    sorted_docs = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
                    return [item["doc"] for item in sorted_docs]


class VectorStoreManager:
    """Manages dense FAISS vector indexing and hybrid Ensemble retrieval."""

    def __init__(
        self,
        embeddings: Embeddings,
        persist_dir: str = "faiss_index",
    ):
        self.embeddings = embeddings
        self.persist_dir = persist_dir
        self.vector_store: Optional[FAISS] = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.indexed_documents: List[Document] = []

    def build_from_documents(
        self,
        documents: List[Document],
        save_to_disk: bool = True,
    ) -> FAISS:
        """Create FAISS and BM25 indices from document chunks."""
        if not documents:
            raise ValueError("Cannot build vector store from empty document list.")

        self.indexed_documents = documents

        # 1. Build FAISS Dense Vector Store
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings,
        )

        # 2. Build BM25 Sparse Keyword Index
        self.bm25_retriever = BM25Retriever.from_documents(documents=documents)

        # 3. Optional persistence
        if save_to_disk:
            self.save_local(self.persist_dir)

        return self.vector_store

    def save_local(self, folder_path: Optional[str] = None) -> None:
        """Save FAISS index to disk."""
        target_dir = folder_path or self.persist_dir
        if self.vector_store is None:
            raise ValueError("No active vector store to save.")
        os.makedirs(target_dir, exist_ok=True)
        self.vector_store.save_local(target_dir)

    def load_local(self, folder_path: Optional[str] = None) -> bool:
        """Load FAISS index from disk if available."""
        target_dir = folder_path or self.persist_dir
        if not os.path.exists(target_dir):
            return False

        try:
            self.vector_store = FAISS.load_local(
                target_dir,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            # Reconstruct BM25 if documents are accessible from docstore
            docs = list(self.vector_store.docstore._dict.values())
            self.indexed_documents = docs
            if docs:
                self.bm25_retriever = BM25Retriever.from_documents(docs)
            return True
        except Exception:
            return False

    def get_retriever(
        self,
        retriever_type: str = "hybrid",
        top_k: int = 4,
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ) -> BaseRetriever:
        """Return configured retriever: 'dense', 'bm25', or 'hybrid' (EnsembleRetriever)."""
        if self.vector_store is None:
            raise ValueError("Vector store has not been built or loaded yet.")

        dense_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": top_k}
        )

        retriever_type = retriever_type.lower().strip()

        if retriever_type == "dense":
            base = dense_retriever

        elif retriever_type == "bm25":
            if self.bm25_retriever is None:
                raise ValueError("BM25 retriever is not initialized.")
            self.bm25_retriever.k = top_k
            base = self.bm25_retriever

        elif retriever_type == "hybrid":
            if self.bm25_retriever is None:
                base = dense_retriever
            else:
                self.bm25_retriever.k = top_k
                base = EnsembleRetriever(
                    retrievers=[dense_retriever, self.bm25_retriever],
                    weights=[dense_weight, bm25_weight],
                )
        else:
            raise ValueError(f"Unknown retriever_type: '{retriever_type}'. Options: ['dense', 'bm25', 'hybrid']")

        # Wrap to strictly enforce top_k limit across union retrievers
        class TopKRetriever(BaseRetriever):
            underlying_retriever: BaseRetriever
            k: int

            def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
                docs = self.underlying_retriever.invoke(query, **kwargs)
                return docs[:self.k]

        return TopKRetriever(underlying_retriever=base, k=top_k)

    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        """Perform dense vector similarity search with similarity distances."""
        if self.vector_store is None:
            raise ValueError("Vector store is not initialized.")
        return self.vector_store.similarity_search_with_score(query, k=k)

    def count_chunks(self) -> int:
        """Return the number of indexed chunks."""
        if self.vector_store is None:
            return 0
        return len(self.vector_store.docstore._dict)

