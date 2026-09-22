"""Document chunker module with metadata tracking and recursive splitting."""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    """Configurable chunker utilizing recursive character splitting."""

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 80,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split a list of documents into overlapping chunks with enriched metadata."""
        if not documents:
            return []

        chunks = self.splitter.split_documents(documents)

        # Enrich chunk metadata with chunk indexing
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = idx
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", 1)
            chunk.metadata["chunk_id"] = f"{source}::p{page}::c{idx}"
            chunk.metadata["char_count"] = len(chunk.page_content)

        return chunks

