"""Streamlit Web UI for RAG-based Document Q&A System with multi-provider support."""

import os
import streamlit as st
from dotenv import load_dotenv

# Load local environment variables
load_dotenv()

from core.document_loader import DocumentLoaderService
from core.chunker import DocumentChunker
from core.embeddings import EmbeddingFactory
from core.vector_store import VectorStoreManager
from core.rag_chain import RAGPipeline, LLMFactory
from core.evaluation import RAGEvaluator

st.set_page_config(
    page_title="RAG Document Q&A Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }
    .citation-card {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 10px 14px;
        margin-top: 8px;
        margin-bottom: 8px;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        background-color: #e2e8f0;
        color: #334155;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- SESSION STATE INITIALIZATION -----------------
if "vector_manager" not in st.session_state:
    st.session_state.vector_manager = None
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0


# ----------------- SIDEBAR CONFIGURATION -----------------
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("1. Providers & Keys")
    llm_provider = st.selectbox(
        "LLM Provider",
        options=["gemini", "openai", "groq", "mock"],
        index=0,
        help="Select generative model provider. 'mock' works offline without API keys.",
    )

    api_key_input = ""
    if llm_provider == "gemini":
        default_gemini_key = os.getenv("GEMINI_API_KEY", "")
        api_key_input = st.text_input(
            "Gemini API Key",
            value=default_gemini_key,
            type="password",
            help="Get a free key from https://aistudio.google.com/app/apikey",
        )
        if not api_key_input:
            st.caption("💡 *Tip: Leave blank & select **mock** above to test offline without an API key.*")
    elif llm_provider == "openai":
        default_openai_key = os.getenv("OPENAI_API_KEY", "")
        api_key_input = st.text_input(
            "OpenAI API Key",
            value=default_openai_key,
            type="password",
            help="Get from OpenAI Platform",
        )
        if not api_key_input:
            st.caption("💡 *Tip: Or select **mock** above to test without an API key.*")
    elif llm_provider == "groq":
        default_groq_key = os.getenv("GROQ_API_KEY", "")
        api_key_input = st.text_input(
            "Groq API Key",
            value=default_groq_key,
            type="password",
        )

    embedding_provider = st.selectbox(
        "Embedding Provider",
        options=["huggingface", "gemini", "openai", "mock"],
        index=0,
        help="'huggingface' runs 100% locally and free with all-MiniLM-L6-v2!",
    )

    st.subheader("2. Retrieval & Chunking")
    retriever_strategy = st.radio(
        "Retrieval Strategy",
        options=["hybrid", "dense"],
        format_func=lambda x: "Hybrid (FAISS Dense + BM25 Sparse)" if x == "hybrid" else "Dense Only (FAISS)",
        index=0,
    )

    top_k = st.slider("Top-K Passages", min_value=1, max_value=8, value=4)
    chunk_size = st.slider("Chunk Size (characters)", min_value=200, max_value=1200, value=600, step=50)
    chunk_overlap = st.slider("Chunk Overlap (characters)", min_value=0, max_value=200, value=80, step=10)

    st.markdown("---")
    col_reset1, col_reset2 = st.columns(2)
    with col_reset1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.rag_pipeline:
                st.session_state.rag_pipeline.clear_history()
            st.rerun()

    with col_reset2:
        if st.button("Reset Index", use_container_width=True):
            st.session_state.vector_manager = None
            st.session_state.rag_pipeline = None
            st.session_state.indexed_files = []
            st.session_state.total_chunks = 0
            st.session_state.messages = []
            st.rerun()


# ----------------- MAIN INTERFACE -----------------
st.markdown('<div class="main-title">📚 RAG Document Q&A Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Retrieval-Augmented Generation over your documents with grounded answers, hybrid search, and source citations.</div>',
    unsafe_allow_html=True,
)

tab_chat, tab_evaluation, tab_about = st.tabs(["💬 Q&A Chat", "📊 Evaluation & Benchmark", "ℹ️ System Info"])

# ----------------- TAB 1: CHAT & INGESTION -----------------
with tab_chat:
    # Ingestion Card
    with st.expander("📂 Ingest Documents (Upload PDFs, TXT, DOCX or Load Samples)", expanded=(st.session_state.vector_manager is None)):
        col_up, col_sample = st.columns([3, 2])

        with col_up:
            uploaded_files = st.file_uploader(
                "Upload document files",
                type=["pdf", "txt", "md", "docx"],
                accept_multiple_files=True,
                help="Supports PDF, TXT, Markdown, and Word DOCX formats",
            )
            btn_index = st.button("🚀 Index Uploaded Documents", type="primary", disabled=not uploaded_files)

        with col_sample:
            st.markdown("**Or quickly test with included sample documents:**")
            st.markdown("- `company_policies.txt` (Remote work, PTO, benefits)")
            st.markdown("- `rag_architecture_overview.txt` (RAG concepts, embeddings)")
            st.markdown("- `sample_ai_paper.pdf` (Transformer self-attention PDF paper)")
            btn_sample = st.button("📖 Load Sample Knowledge Base")

        # Handle indexing action
        if btn_index or btn_sample:
            with st.spinner("Processing, chunking, and embedding documents..."):
                try:
                    all_docs = []
                    file_names = []

                    if btn_sample:
                        sample_dir = os.path.join(os.path.dirname(__file__), "data", "sample_docs")
                        sample_paths = [
                            os.path.join(sample_dir, "company_policies.txt"),
                            os.path.join(sample_dir, "rag_architecture_overview.txt"),
                            os.path.join(sample_dir, "sample_ai_paper.pdf"),
                        ]
                        for path in sample_paths:
                            if os.path.exists(path):
                                docs = DocumentLoaderService.load_from_file_path(path)
                                all_docs.extend(docs)
                                file_names.append(os.path.basename(path))
                    else:
                        for uf in uploaded_files:
                            content = uf.read()
                            docs = DocumentLoaderService.load_from_bytes(content, uf.name)
                            all_docs.extend(docs)
                            file_names.append(uf.name)

                    if not all_docs:
                        st.error("No valid text could be extracted from the specified documents.")
                    else:
                        # 1. Chunk documents
                        chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                        chunks = chunker.split_documents(all_docs)

                        # 2. Initialize Embeddings
                        emb = EmbeddingFactory.get_embeddings(
                            provider=embedding_provider,
                            api_key=api_key_input if api_key_input else None,
                        )

                        # 3. Build Vector Store
                        v_mgr = VectorStoreManager(embeddings=emb)
                        v_mgr.build_from_documents(chunks, save_to_disk=False)

                        # 4. Initialize LLM & Pipeline
                        llm = LLMFactory.get_llm(
                            provider=llm_provider,
                            api_key=api_key_input if api_key_input else None,
                        )
                        retriever = v_mgr.get_retriever(retriever_type=retriever_strategy, top_k=top_k)
                        pipeline = RAGPipeline(retriever=retriever, llm=llm)

                        st.session_state.vector_manager = v_mgr
                        st.session_state.rag_pipeline = pipeline
                        st.session_state.indexed_files = file_names
                        st.session_state.total_chunks = len(chunks)

                        st.success(f"Successfully indexed {len(file_names)} documents into {len(chunks)} searchable chunks!")
                        st.rerun()

                except Exception as e:
                    st.error(f"Error during ingestion: {str(e)}")

    # Index Status Banner
    if st.session_state.vector_manager is not None:
        st.info(
            f"✅ **Knowledge Base Ready**: {len(st.session_state.indexed_files)} files indexed "
            f"({st.session_state.total_chunks} chunks). Strategy: **{retriever_strategy.upper()}** (Top-{top_k})"
        )
    else:
        st.warning("⚠️ No documents indexed yet. Please upload documents or click 'Load Sample Knowledge Base' above.")

    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander(f"📑 Sources Cited ({len(msg['sources'])} passages)", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(
                            f"**Passage {src['passage_index']}** • `{src['source']}` (Page {src['page']}) "
                            f"<span class='badge'>{src['chunk_id']}</span>",
                            unsafe_allow_html=True,
                        )
                        st.caption(f'"{src["snippet"]}"')

    # Chat Input Box
    user_query = st.chat_input("Ask a question about the indexed documents...")

    if user_query:
        if st.session_state.rag_pipeline is None:
            st.error("Please index documents before asking questions.")
        else:
            # Display user message
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            # Generate assistant response
            with st.chat_message("assistant"):
                with st.spinner("Searching passages and synthesizing answer..."):
                    try:
                        # 1. Dynamically sync retriever in case user modified strategy or top_k
                        updated_retriever = st.session_state.vector_manager.get_retriever(
                            retriever_type=retriever_strategy,
                            top_k=top_k,
                        )
                        st.session_state.rag_pipeline.retriever = updated_retriever

                        # 2. Dynamically sync LLM in case user changed provider or key in sidebar
                        updated_llm = LLMFactory.get_llm(
                            provider=llm_provider,
                            api_key=api_key_input if api_key_input else None,
                        )
                        from langchain_core.output_parsers import StrOutputParser
                        st.session_state.rag_pipeline.llm = updated_llm
                        st.session_state.rag_pipeline.chain = (
                            st.session_state.rag_pipeline.prompt_template
                            | updated_llm
                            | StrOutputParser()
                        )

                        res = st.session_state.rag_pipeline.query(user_query, include_history=True)
                        answer = res["answer"]
                        sources = res["sources"]

                        st.markdown(answer)

                        if sources:
                            with st.expander(f"📑 Sources Cited ({len(sources)} passages)", expanded=True):
                                for src in sources:
                                    st.markdown(
                                        f"**Passage {src['passage_index']}** • `{src['source']}` (Page {src['page']}) "
                                        f"<span class='badge'>{src['chunk_id']}</span>",
                                        unsafe_allow_html=True,
                                    )
                                    st.caption(f'"{src["snippet"]}"')

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "sources": sources,
                            }
                        )

                    except Exception as err:
                        err_str = str(err)
                        if any(k in err_str for k in ["API_KEY_INVALID", "API key not valid", "GEMINI_API_KEY"]):
                            st.error(
                                "🔑 **Gemini API Key Required or Invalid**\n\n"
                                "Your query could not be sent to Google Gemini because a valid API key is needed.\n\n"
                                "**How to fix:**\n"
                                "1. **Enter a Gemini API Key**: Paste your key into the **Gemini API Key** field in the left sidebar (get one free at [Google AI Studio](https://aistudio.google.com/app/apikey)).\n"
                                "2. **Or Test Offline for Free**: In the left sidebar, change **LLM Provider** from `gemini` to `mock` to test the entire retrieval and citation pipeline immediately without any API key."
                            )
                        elif "OPENAI_API_KEY" in err_str:
                            st.error(
                                "🔑 **OpenAI API Key Missing or Invalid**\n\n"
                                "Please paste your OpenAI API key in the sidebar or switch **LLM Provider** to `mock`."
                            )
                        elif "GROQ_API_KEY" in err_str:
                            st.error(
                                "🔑 **Groq API Key Missing or Invalid**\n\n"
                                "Please paste your Groq API key in the sidebar or switch **LLM Provider** to `mock`."
                            )
                        else:
                            st.error(f"Generation error: {err_str}")


# ----------------- TAB 2: EVALUATION & BENCHMARK -----------------
with tab_evaluation:
    st.subheader("📈 Grounding & Retrieval Benchmark")
    st.markdown(
        """
        Evaluate the RAG pipeline against groundedness, answer relevance, and compare **Dense vs. Hybrid (BM25 + Dense)** retrieval performance.
        """
    )

    if st.session_state.vector_manager is None:
        st.warning("Please index documents in the 'Q&A Chat' tab first.")
    else:
        st.markdown("#### 1. Compare Dense vs. Hybrid Retrieval")
        sample_benchmark_queries = [
            {"query": "How many days per week can employees work remotely?", "keyword": "remotely"},
            {"query": "What is the maximum reimbursement for home office equipment?", "keyword": "$1,200"},
            {"query": "What hardware refreshes are provided for engineering laptops?", "keyword": "refreshed"},
            {"query": "Why does RAG prevent LLM hallucination?", "keyword": "hallucination"},
            {"query": "What are optimal chunk sizes for technical document retrieval?", "keyword": "overlap"},
        ]

        if st.button("⚡ Run Dense vs. Hybrid Retrieval Benchmark"):
            with st.spinner("Benchmarking retrieval strategies across test queries..."):
                bench_res = RAGEvaluator.compare_retrievers(
                    st.session_state.vector_manager,
                    sample_benchmark_queries,
                    top_k=top_k,
                )

                col_b1, col_b2, col_b3 = st.columns(3)
                col_b1.metric("Queries Tested", bench_res["total_queries"])
                col_b2.metric("Avg Dense Latency", f"{bench_res['avg_dense_latency_ms']} ms")
                col_b3.metric("Avg Hybrid Latency", f"{bench_res['avg_hybrid_latency_ms']} ms")

                st.dataframe(bench_res["records"], use_container_width=True)

        st.markdown("---")
        st.markdown("#### 2. Pipeline Groundedness & Faithfulness Audit")
        if st.button("🧪 Run Groundedness Audit"):
            with st.spinner("Evaluating faithfulness and answer relevance..."):
                audit_tests = [
                    {"question": "What is the parental leave policy?"},
                    {"question": "What is the daily meal allowance when traveling?"},
                    {"question": "What happens if a question asks about unmentioned policies like stock options?"},
                ]
                eval_res = RAGEvaluator.evaluate_pipeline(st.session_state.rag_pipeline, audit_tests)

                col_e1, col_e2, col_e3 = st.columns(3)
                col_e1.metric("Avg Groundedness Score", f"{eval_res['average_groundedness'] * 100:.1f}%")
                col_e2.metric("Avg Answer Relevance", f"{eval_res['average_relevance'] * 100:.1f}%")
                col_e3.metric("Avg Latency", f"{eval_res['average_latency_sec']} s")

                st.json(eval_res["detailed_results"])


# ----------------- TAB 3: SYSTEM INFO -----------------
with tab_about:
    st.subheader("ℹ️ System Architecture & Technology Stack")
    st.markdown(
        """
        - **Orchestration**: LangChain (LCEL Expression Language)
        - **Embedding Options**:
          - `sentence-transformers/all-MiniLM-L6-v2` (Local, zero cost)
          - Google `text-embedding-004`
          - OpenAI `text-embedding-3-small`
        - **Vector Store**: FAISS (Facebook AI Similarity Search)
        - **Sparse Search**: BM25 (Rank-BM25)
        - **Hybrid Retrieval**: LangChain `EnsembleRetriever` with Convex Rank Fusion
        - **LLMs**: Google Gemini 1.5 Flash, OpenAI GPT-4o-mini, Groq LLaMA 3.1
        - **Document Loaders**: PyPDF, Docx2txt, UTF-8 TextLoader
        - **Anti-Hallucination**: Grounded system prompt with explicit abstention & citation extraction
        """
    )

