"""Embedding factory providing HuggingFace (local), Google Gemini, and OpenAI embeddings."""

import os
from typing import Optional
from langchain_core.embeddings import Embeddings


class EmbeddingFactory:
    """Factory for instantiating document embedding models."""

    @staticmethod
    def get_embeddings(
        provider: str = "huggingface",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> Embeddings:
        """Instantiate and return an embedding model based on provider."""
        provider = provider.lower().strip()

        if provider in {"huggingface", "local", "sentence-transformers"}:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            model = model_name or "sentence-transformers/all-MiniLM-L6-v2"
            return HuggingFaceEmbeddings(
                model_name=model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

        elif provider in {"gemini", "google"}:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not key:
                raise ValueError(
                    "Google Gemini API key not found. Please set GEMINI_API_KEY in your environment or .env file."
                )
            model = model_name or "models/text-embedding-004"
            return GoogleGenerativeAIEmbeddings(
                model=model,
                google_api_key=key,
            )

        elif provider in {"openai"}:
            from langchain_openai import OpenAIEmbeddings

            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError(
                    "OpenAI API key not found. Please set OPENAI_API_KEY in your environment or .env file."
                )
            model = model_name or "text-embedding-3-small"
            return OpenAIEmbeddings(
                model=model,
                openai_api_key=key,
            )

        elif provider in {"mock", "fake"}:
            from langchain_community.embeddings import FakeEmbeddings

            return FakeEmbeddings(size=384)

        else:
            raise ValueError(
                f"Unsupported embedding provider: '{provider}'. Supported: ['huggingface', 'gemini', 'openai', 'mock']"
            )

