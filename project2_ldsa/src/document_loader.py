"""Load Moroccan legal PDFs and text files with language metadata."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from src.config import DATA_DIR

logger = logging.getLogger(__name__)
SUPPORTED_LANGUAGES = {"ar", "fr"}
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


class LegalDocumentLoader:
    """Load documents from data/morocco/ar and data/morocco/fr."""

    def __init__(self, data_path: str | Path = DATA_DIR) -> None:
        self.data_path = Path(data_path)
        if not self.data_path.is_dir():
            raise FileNotFoundError(f"Corpus directory not found: {self.data_path}")

    def load_documents(self) -> list[Document]:
        files = sorted(
            path
            for path in self.data_path.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not files:
            raise ValueError(f"No PDF or TXT documents found in {self.data_path}")

        documents: list[Document] = []
        for path in files:
            relative_path = path.relative_to(self.data_path)
            language = relative_path.parts[0].lower()
            if language not in SUPPORTED_LANGUAGES:
                logger.warning("Skipping %s: expected an ar/ or fr/ directory", path)
                continue

            if path.suffix.lower() == ".pdf":
                loaded = PyPDFLoader(str(path)).load()
            else:
                loaded = TextLoader(str(path), encoding="utf-8").load()

            for document in loaded:
                document.metadata.update(
                    {
                        "source": str(path),
                        "language": language,
                        "jurisdiction": "MA",
                        "document_type": "moroccan_legislation",
                    }
                )
                documents.append(document)

        if not documents:
            raise ValueError("No readable Arabic/French legal documents were loaded")

        logger.info("Loaded %d pages from %d files", len(documents), len(files))
        return documents
