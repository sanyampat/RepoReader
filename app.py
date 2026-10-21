from __future__ import annotations

import hashlib
import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from repo_rag.ingest import zip_to_documents
from repo_rag.graph import build_graph

st.set_page_config(page_title="RepoRAG", page_icon="🧠", layout="wide")
st.title("🧠 RepoRAG")
st.caption("Ask questions about a codebase. Answers are grounded in retrieved source code.")

with st.sidebar:
    st.header("Local model settings")
    chat_model = st.text_input("Ollama chat model", value="qwen2.5-coder:3b")
    embedding_model = st.text_input("Ollama embedding model", value="nomic-embed-text")
    st.caption("Both models must be available in your local Ollama installation.")
    st.divider()
    st.markdown("**Indexed repositories are stored locally** in `.chroma/`.")

uploaded = st.file_uploader("Upload repository ZIP", type=["zip"])
if uploaded is not None:
    zip_bytes = uploaded.getvalue()
    repo_hash = hashlib.sha256(zip_bytes).hexdigest()[:16]
    if (st.session_state.get("repo_hash") != repo_hash
            or st.session_state.get("indexed_embedding_model") != embedding_model):
        status_placeholder = st.empty()
        status_placeholder.info("Reading source files and creating code chunks...")
        try:
            chunks, stats = zip_to_documents(zip_bytes)
            # Unique collection per ZIP, so uploads do not mix together.
            collection = f"repo_{repo_hash}"
            embeddings = OllamaEmbeddings(model=embedding_model)
            store = Chroma(
                collection_name=collection,
                embedding_function=embeddings,
                persist_directory=".chroma",
            )
            # Rebuild this collection on a fresh indexing run.
            existing = store.get()
            if existing.get("ids"):
                store.delete(ids=existing["ids"])

            total_chunks = len(chunks)
            ids = [f"{repo_hash}_{i}" for i in range(total_chunks)]

            # Batch documents to avoid overloading Ollama's internal runner process
            batch_size = 64
            progress_bar = st.progress(0.0)
            progress_text = st.empty()

            for start_idx in range(0, total_chunks, batch_size):
                end_idx = min(start_idx + batch_size, total_chunks)
                batch_chunks = chunks[start_idx:end_idx]
                batch_ids = ids[start_idx:end_idx]

                try:
                    store.add_documents(batch_chunks, ids=batch_ids)
                except Exception as batch_exc:
                    # Fallback to smaller sub-batches if the runner had a hiccup
                    sub_batch_size = 16
                    for s_idx in range(0, len(batch_chunks), sub_batch_size):
                        e_idx = min(s_idx + sub_batch_size, len(batch_chunks))
                        store.add_documents(
                            batch_chunks[s_idx:e_idx],
                            ids=batch_ids[s_idx:e_idx],
                        )

                progress = end_idx / total_chunks
                progress_bar.progress(progress)
                progress_text.caption(f"Generating embeddings: {end_idx}/{total_chunks} chunks ({int(progress * 100)}%)...")

            progress_bar.empty()
            progress_text.empty()
            status_placeholder.empty()

            st.session_state["repo_hash"] = repo_hash
            st.session_state["collection"] = collection
            st.session_state["stats"] = stats
            st.session_state["embedding_model"] = embedding_model
            st.session_state["indexed_embedding_model"] = embedding_model
            st.session_state.pop("graph_cache", None)
            st.session_state.pop("graph_cache_key", None)
            st.session_state.pop("graph_html_key", None)
        except Exception as exc:
            status_placeholder.empty()
            st.error(f"Indexing failed: {exc}")
            st.info("Check that Ollama is running and the embedding model is installed.")
            st.stop()

    stats = st.session_state.get("stats", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Files indexed", stats.get("files_indexed", 0))
    c2.metric("Code chunks", stats.get("chunks_created", 0))
    c3.metric("Files skipped", stats.get("files_skipped", 0))

    st.success("Repository indexed.")
    chat_tab, graph_tab = st.tabs(["💬 Ask RepoRAG", "🕸️ Knowledge Graph"])

    with chat_tab:
        question = st.chat_input("e.g. How does authentication work?")
        if question:
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                with st.spinner("Searching the codebase and generating an answer..."):
                    try:
                        cache_key = (st.session_state["collection"], chat_model, embedding_model)
                        if st.session_state.get("graph_cache_key") != cache_key:
                            graph, _ = build_graph(st.session_state["collection"], chat_model, embedding_model)
                            st.session_state["graph_cache"] = graph
                            st.session_state["graph_cache_key"] = cache_key
                        result = st.session_state["graph_cache"].invoke({"question": question})
                        st.markdown(result.get("answer", "No answer returned."))
                        sources = result.get("sources", [])
                        if sources:
                            with st.expander(f"Retrieved source chunks ({len(sources)})"):
                                for source in sources:
                                    st.markdown(f'**`{source["path"]}` — L{source["start_line"]}-L{source["end_line"]}**')
                                    st.code(source["code"], language="text")
                    except Exception as exc:
                        st.error(f"Question failed: {exc}")
                        st.info("Verify that the chat and embedding models are installed in Ollama, and that the model names match `ollama list`.")

    with graph_tab:
        st.markdown("### Repository connections")
        st.caption("Arrows represent local import relationships detected in source files. Hover over nodes to inspect paths and connection counts.")
        try:
            from streamlit.components.v1 import html as render_html
            from repo_rag.graph_view import build_repository_graph
            graph_key = f"repo_graph_{repo_hash}"
            if st.session_state.get("graph_html_key") != graph_key:
                graph_html, graph_stats = build_repository_graph(zip_bytes)
                st.session_state["graph_html"] = graph_html
                st.session_state["graph_stats"] = graph_stats
                st.session_state["graph_html_key"] = graph_key
            graph_stats = st.session_state["graph_stats"]
            a, b, c = st.columns(3)
            a.metric("Files", graph_stats["files"])
            b.metric("Import connections", graph_stats["connections"])
            c.metric("Isolated files", graph_stats["isolated"])
            if graph_stats["connections"] == 0:
                st.warning("No local import connections detected. The repository may use unsupported import styles.")
            render_html(st.session_state["graph_html"], height=700, scrolling=False)
        except Exception as exc:
            st.error(f"Could not build the repository graph: {exc}")

else:
    st.info("Upload a ZIP of a source-code repository to begin.")
    st.markdown("""
    **Good first test:** a small Python or React project.

    The indexer skips common dependency/build folders and secret-like files, and it does
    not extract ZIP paths onto disk. Do not upload repositories containing secrets.
    """)
