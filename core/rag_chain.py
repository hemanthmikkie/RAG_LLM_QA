"""RAG QA Chain using LangChain LCEL with grounding, multi-turn memory, and source citations."""

import os
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever


def format_context_with_citations(documents: List[Document]) -> str:
    """Format retrieved document chunks into clean indexed context passages."""
    formatted_parts = []
    for idx, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page", 1)
        chunk_id = doc.metadata.get("chunk_id", f"c{idx}")
        header = f"--- [Passage {idx} | Source: {source} | Page: {page} | ID: {chunk_id}] ---"
        formatted_parts.append(f"{header}\n{doc.page_content.strip()}")
    return "\n\n".join(formatted_parts)


def extract_citations(documents: List[Document]) -> List[Dict[str, Any]]:
    """Extract clean metadata and text snippets for UI/API citation displays."""
    citations = []
    for idx, doc in enumerate(documents, start=1):
        content = doc.page_content.strip()
        # Create a concise snippet of first 200 characters
        snippet = content[:200] + "..." if len(content) > 200 else content
        citations.append(
            {
                "passage_index": idx,
                "source": doc.metadata.get("source", "Unknown Document"),
                "page": doc.metadata.get("page", 1),
                "chunk_id": doc.metadata.get("chunk_id", f"c{idx}"),
                "snippet": snippet,
                "full_text": content,
            }
        )
    return citations


class LLMFactory:
    """Factory for instantiating LLM chat models across providers."""

    @staticmethod
    def get_llm(
        provider: str = "gemini",
        model_name: Optional[str] = None,
        temperature: float = 0.2,
        api_key: Optional[str] = None,
    ) -> BaseChatModel:
        """Instantiate chat model from provider name."""
        provider = provider.lower().strip()

        if provider in {"gemini", "google"}:
            from langchain_google_genai import ChatGoogleGenerativeAI

            key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not key:
                raise ValueError("GEMINI_API_KEY must be configured to use Google Gemini models.")
            model = model_name or "gemini-1.5-flash"
            return ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                google_api_key=key,
            )

        elif provider in {"openai"}:
            from langchain_openai import ChatOpenAI

            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError("OPENAI_API_KEY must be configured to use OpenAI models.")
            model = model_name or "gpt-4o-mini"
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                openai_api_key=key,
            )

        elif provider in {"groq"}:
            from langchain_community.chat_models import ChatGroq

            key = api_key or os.getenv("GROQ_API_KEY")
            if not key:
                raise ValueError("GROQ_API_KEY must be configured to use Groq models.")
            model = model_name or "llama-3.1-8b-instant"
            return ChatGroq(
                model_name=model,
                temperature=temperature,
                groq_api_key=key,
            )

        elif provider in {"mock", "fake"}:
            import re
            from langchain_core.language_models.chat_models import SimpleChatModel

            class MockChatModel(SimpleChatModel):
                @property
                def _llm_type(self) -> str:
                    return "mock"

                def _call(self, messages, stop=None, run_manager=None, **kwargs):
                    question = ""
                    context = ""
                    for m in messages:
                        content = getattr(m, "content", "")
                        if hasattr(m, "__class__") and "Human" in m.__class__.__name__:
                            question = content
                        elif "Context:" in content:
                            context = content.split("Context:", 1)[1]

                    if question and context:
                        words = [w.lower().strip(".,;:?!'\"()[]") for w in question.split() if len(w) > 3]
                        sentences = [
                            s.strip()
                            for s in re.split(r"(?<=[.!?])\s+|\n+", context)
                            if len(s.strip()) > 15 and not s.strip().startswith("---")
                        ]
                        scored = []
                        for s in sentences:
                            s_lower = s.lower()
                            hits = sum(1 for w in words if w in s_lower)
                            if hits > 0:
                                scored.append((hits, s))
                        scored.sort(key=lambda x: x[0], reverse=True)

                        if scored:
                            top_snippets = [f"• {item[1]}" for item in scored[:3]]
                            return (
                                "**[Offline Mode - Extractive Grounding]**\n\n"
                                + "\n\n".join(top_snippets)
                                + "\n\n*(Tip: Add a Gemini API key in the sidebar for conversational generative answers).* "
                            )

                    return "This is a mock grounded answer based on the provided context."

            return MockChatModel()

        else:
            raise ValueError(f"Unsupported LLM provider: '{provider}'")


SYSTEM_PROMPT = """You are a precise, helpful, and strictly grounded Document Q&A assistant.
Your task is to answer the user's question based ONLY on the provided context passages below.

Context:
{context}

Guidelines:
1. Ground your answer strictly in the provided Context. Do NOT use outside knowledge or hallucinate facts.
2. If the Context does not provide sufficient information to answer the question, explicitly state:
   "I cannot answer this question based on the provided documents as the necessary information is not present."
3. Cite the relevant Passage Numbers (e.g., [Passage 1], [Passage 2]) or document sources in your explanation where appropriate.
4. Keep the answer direct, well-structured, and factual.
"""


class RAGPipeline:
    """Conversational RAG Pipeline handling context retrieval, history, and generation."""

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseChatModel,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        self.retriever = retriever
        self.llm = llm
        self.system_prompt = system_prompt
        self.chat_history: List[BaseMessage] = []

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )

        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def query(
        self,
        question: str,
        include_history: bool = True,
    ) -> Dict[str, Any]:
        """Execute RAG query, returning answer and verified source citations."""
        if not question or not question.strip():
            raise ValueError("Query string cannot be empty.")

        # 1. Retrieve top relevant chunks
        retrieved_docs: List[Document] = self.retriever.invoke(question)

        # 2. Format context and citations
        formatted_context = format_context_with_citations(retrieved_docs)
        citations = extract_citations(retrieved_docs)

        # 3. Handle conversation history
        history_to_send = self.chat_history if include_history else []

        # 4. Generate answer through LCEL chain
        answer = self.chain.invoke(
            {
                "context": formatted_context,
                "chat_history": history_to_send,
                "question": question,
            }
        )

        # 5. Update history if conversational
        if include_history:
            self.chat_history.append(HumanMessage(content=question))
            self.chat_history.append(AIMessage(content=answer))

        return {
            "question": question,
            "answer": answer,
            "sources": citations,
            "num_passages_retrieved": len(retrieved_docs),
            "raw_context": formatted_context,
        }

    def clear_history(self) -> None:
        """Clear conversation memory."""
        self.chat_history = []

