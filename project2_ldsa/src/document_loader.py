"""Load Moroccan legal PDFs and text files with language/source metadata."""
from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from src.config import DATA_DIR

logger = logging.getLogger(__name__)


class LegalDocumentLoader:
    """Load PDF/TXT documents from data/morocco/{ar,fr}/."""

    def __init__(self, data_path: str | Path = DATA_DIR):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path '{self.data_path}' does not exist")

    def load_documents(self) -> list[Document]:
        files = sorted(
            path for path in self.data_path.rglob("*")
            if path.is_file() and path.suffix.lower() in {".pdf", ".txt"}
        )
        if not files:
            raise ValueError(f"No PDF or TXT legal documents found in {self.data_path}")

        documents: list[Document] = []
        for path in files:
            language = path.relative_to(self.data_path).parts[0].lower()
            if language not in {"ar", "fr"}:
                logger.warning("Skipping %s: put documents under ar/ or fr/", path)
                continue
            try:
                if path.suffix.lower() == ".pdf":
                    loaded = PyPDFLoader(str(path)).load()
                else:
                    loaded = TextLoader(str(path), encoding="utf-8").load()
                for doc in loaded:
                    doc.metadata.update({
                        "source": str(path),
                        "language": language,
                        "jurisdiction": "MA",
                        "document_type": "moroccan_legislation",
                    })
                    documents.append(doc)
            except Exception:
                logger.exception("Could not load %s", path)
                raise

        if not documents:
            raise ValueError("No readable Arabic/French legal documents were loaded")
        logger.info("Loaded %d pages/documents from %d files", len(documents), len(files))
        return documents
