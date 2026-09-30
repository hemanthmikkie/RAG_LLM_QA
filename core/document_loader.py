"""Document loader module supporting PDF, TXT, MD, and DOCX formats with metadata preservation."""

import os
import re
import tempfile
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)


def clean_text(text: str) -> str:
    """Normalize whitespace and strip spurious control characters."""
    if not text:
        return ""
    # Normalize unicode whitespace and linebreaks
    text = re.sub(r"\r\n|\r", "\n", text)
    # Collapse multiple consecutive empty lines to two
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse consecutive horizontal spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


class DocumentLoaderService:
    """Unified document loader service for multi-format ingestion."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}

    @classmethod
    def load_from_file_path(cls, file_path: str) -> List[Document]:
        """Load and clean documents from a local filesystem path."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format '{ext}'. Supported: {cls.SUPPORTED_EXTENSIONS}"
            )

        filename = os.path.basename(file_path)

        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
            raw_docs = loader.load()
        elif ext in {".txt", ".md"}:
            loader = TextLoader(file_path, encoding="utf-8")
            raw_docs = loader.load()
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
            raw_docs = loader.load()
        else:
            raise ValueError(f"No loader configured for {ext}")

        cleaned_docs = []
        for doc in raw_docs:
            cleaned_content = clean_text(doc.page_content)
            if cleaned_content:
                metadata = dict(doc.metadata)
                metadata["source"] = filename
                metadata["file_path"] = file_path
                # Normalize 0-indexed page numbers from PDF parsers to 1-indexed for human readability
                raw_page = metadata.get("page", 0)
                try:
                    metadata["page"] = int(raw_page) + 1 if isinstance(raw_page, int) else 1
                except (ValueError, TypeError):
                    metadata["page"] = 1
                cleaned_docs.append(
                    Document(page_content=cleaned_content, metadata=metadata)
                )

        return cleaned_docs

    @classmethod
    def load_from_bytes(
        cls, file_bytes: bytes, filename: str
    ) -> List[Document]:
        """Load documents from in-memory byte streams (e.g. Streamlit or FastAPI uploads)."""
        ext = os.path.splitext(filename)[1].lower()
        if ext not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format '{ext}'. Supported: {cls.SUPPORTED_EXTENSIONS}"
            )

        # Write temporarily to disk for standard loaders
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            docs = cls.load_from_file_path(tmp_path)
            # Re-tag original filename
            for doc in docs:
                doc.metadata["source"] = filename
                doc.metadata["file_path"] = filename
            return docs
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

