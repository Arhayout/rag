"""Streamlit UI for the Legal Document Search Assistant."""

from __future__ import annotations

import logging

import streamlit as st
from dotenv import load_dotenv

from src.config import DATA_DIR, DEFAULT_TOP_K, VECTOR_STORE_DIR
from src.document_loader import LegalDocumentLoader
from src.vector_store import LegalDocumentStore

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("legal_rag")


def log_error(exc: Exception) -> str:
    logger.error(exc, exc_info=True)
    return f"Error: {exc}"


@st.cache_resource(show_spinner="Initializing retrieval system...")
def initialize_retrieval_system():
    vectorstore = LegalDocumentStore.load_local(VECTOR_STORE_DIR)
    if vectorstore is not None:
        return vectorstore

    loader = LegalDocumentLoader(DATA_DIR)
    documents = loader.load_documents()
    return LegalDocumentStore(vecstore_path=VECTOR_STORE_DIR).save_index(documents)


def render_search_results(vectorstore, query: str, top_k: int) -> None:
    results = LegalDocumentStore.query_similar(vectorstore, query, top_k)
    if not results:
        st.warning("No matching documents found.")
        return

    for doc in results:
        source = doc.metadata.get("source", "unknown")
        with st.expander(f"{source}: {doc.page_content[:80]}..."):
            st.markdown(f"**Summary:** {doc.page_content}")
            st.markdown("**Full document:**")
            st.write(doc.metadata.get("full_content", "No content available"))


def main() -> None:
    st.set_page_config(page_title="Legal Document Finder", page_icon="⚖️")
    st.title("Legal Document Finder")
    st.caption("Semantic search over legal case summaries using LangChain + FAISS")

    try:
        retriever = initialize_retrieval_system()
    except Exception as exc:
        st.error(log_error(exc))
        st.info(
            "Ensure the `data/` folder contains `.txt` legal documents and "
            "dependencies are installed (`pip install -r requirements.txt`)."
        )
        return

    query = st.text_input(
        "Enter your topic or question:",
        placeholder="Copyright disputes in digital media",
    )
    top_k = st.number_input(
        "Number of documents",
        min_value=1,
        max_value=10,
        value=DEFAULT_TOP_K,
    )

    if st.button("Search", type="primary") and query:
        with st.spinner("Searching..."):
            try:
                render_search_results(retriever, query, int(top_k))
            except Exception as exc:
                st.error(log_error(exc))


if __name__ == "__main__":
    main()