"""FAISS vector store for legal document search."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import VECTOR_STORE_DIR
from src.embeddings import get_embeddings
from src.summarizer import generate_summary

logger = logging.getLogger(__name__)


class LegalDocumentStore:
    """Build, persist, and query a FAISS index over legal document summaries."""

    def __init__(self, vecstore_path: str | Path = VECTOR_STORE_DIR):
        self.vectorstore_path = Path(vecstore_path)
        self.embeddings = get_embeddings()

    def generate_summary(self, document: str) -> str:
        return generate_summary(document)

    def create_summary_documents(self, documents: list[Document]) -> list[Document]:
        summary_docs = []
        for doc in documents:
            summary = self.generate_summary(doc.page_content)
            summary_docs.append(
                Document(
                    page_content=summary,
                    metadata={
                        **doc.metadata,
                        "full_content": doc.page_content,
                        "source": doc.metadata.get("source", ""),
                    },
                )
            )
        return summary_docs

    def save_index(self, documents: list[Document]) -> FAISS:
        if not documents:
            raise ValueError("Cannot build index from an empty document list")

        summary_docs = self.create_summary_documents(documents)
        vectorstore = FAISS.from_documents(summary_docs, self.embeddings)
        self.vectorstore_path.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(self.vectorstore_path))
        logger.info("Saved vector store to %s", self.vectorstore_path)
        return vectorstore

    @classmethod
    def load_local(cls, vecstore_path: str | Path = VECTOR_STORE_DIR) -> FAISS | None:
        path = Path(vecstore_path)
        if not path.exists():
            logger.warning("Vector store path does not exist: %s", path)
            return None

        try:
            instance = cls(path)
            return FAISS.load_local(
                str(path),
                instance.embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception as exc:
            logger.error("Failed to load vector store: %s", exc)
            return None

    @staticmethod
    def query_similar(vectorstore: FAISS, query: str, k: int) -> list[Document]:
        if not query.strip():
            raise ValueError("Query must not be empty")
        if k < 1:
            raise ValueError("k must be at least 1")
        return vectorstore.similarity_search(query, k=k)