"""Load legal PDFs and text files with jurisdiction and language metadata."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from src.config import DATA_DIR, JURISDICTIONS

logger = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


class LegalDocumentLoader:
    """Load a jurisdiction-specific corpus organised by language subfolders."""

    def __init__(
        self,
        data_path: str | Path = DATA_DIR,
        jurisdiction: str = "MA",
    ) -> None:
        self.data_path = Path(data_path)
        self.jurisdiction = jurisdiction.upper()
        if self.jurisdiction not in JURISDICTIONS:
            raise ValueError(f"Unsupported jurisdiction: {jurisdiction}")
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

        jurisdiction_config = JURISDICTIONS[self.jurisdiction]
        supported_languages = set(jurisdiction_config["languages"])
        documents: list[Document] = []
        for path in files:
            relative_path = path.relative_to(self.data_path)
            language = relative_path.parts[0].lower()
            if language not in supported_languages:
                logger.warning(
                    "Skipping %s: expected one of %s",
                    path,
                    ", ".join(sorted(supported_languages)),
                )
                continue

            loaded = (
                PyPDFLoader(str(path)).load()
                if path.suffix.lower() == ".pdf"
                else TextLoader(str(path), encoding="utf-8").load()
            )
            for document in loaded:
                document.metadata.update(
                    {
                        "source": str(path),
                        "language": language,
                        "jurisdiction": self.jurisdiction,
                        "document_type": jurisdiction_config["document_type"],
                    }
                )
                documents.append(document)

        if not documents:
            raise ValueError(
                f"No readable {self.jurisdiction} legal documents were loaded"
            )

        logger.info("Loaded %d pages from %d files", len(documents), len(files))
        return documents
