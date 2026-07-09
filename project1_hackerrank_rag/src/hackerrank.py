"""Vector index and retrieval for HackerRank skills report PDFs."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader

from src.config import DATA_DIR, DEFAULT_TOP_K, EMBEDDING_MODEL, INDEX_DIR

logger = logging.getLogger(__name__)

_query_engine = None
_index = None


class DataNotFoundError(FileNotFoundError):
    """Raised when no PDF documents are available for indexing."""


def _configure_settings() -> None:
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)


def _load_documents(data_dir: Path):
    if not data_dir.exists() or not list(data_dir.glob("*.pdf")):
        raise DataNotFoundError(
            f"No PDF files found in {data_dir}. "
            "Run scripts/download_data.sh or add PDFs manually."
        )

    return SimpleDirectoryReader(
        str(data_dir),
        required_exts=[".pdf"],
        file_extractor={".pdf": PyMuPDFReader()},
    ).load_data()


def get_index(documents, index_name: str | Path) -> VectorStoreIndex:
    index_path = str(index_name)
    if not os.path.exists(index_path):
        logger.info("Building index at %s", index_path)
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        index.storage_context.persist(persist_dir=index_path)
        return index

    logger.info("Loading existing index from %s", index_path)
    return load_index_from_storage(
        StorageContext.from_defaults(persist_dir=index_path)
    )


def get_index_instance() -> VectorStoreIndex:
    global _index
    if _index is not None:
        return _index

    _configure_settings()
    documents = _load_documents(DATA_DIR)
    _index = get_index(documents, INDEX_DIR)
    return _index


def get_query_engine():
    global _query_engine
    if _query_engine is not None:
        return _query_engine

    _query_engine = get_index_instance().as_query_engine(similarity_top_k=DEFAULT_TOP_K)
    return _query_engine


def retrieve_chunks(question: str, top_k: int = DEFAULT_TOP_K):
    retriever = get_index_instance().as_retriever(similarity_top_k=top_k)
    return retriever.retrieve(question)