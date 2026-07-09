"""Load legal case text files from the data directory."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

from src.config import DATA_DIR

logger = logging.getLogger(__name__)


class LegalDocumentLoader:
    """Load `.txt` legal documents from a directory."""

    def __init__(self, data_path: str | Path = DATA_DIR):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path '{self.data_path}' does not exist")

    def load_documents(self) -> list[Document]:
        try:
            loader = DirectoryLoader(
                str(self.data_path),
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"autodetect_encoding": True},
                show_progress=False,
            )
            loaded = loader.load()
            if not loaded:
                raise ValueError(f"No .txt documents found in {self.data_path}")

            docs: list[Document] = []
            for doc in loaded:
                source = os.path.basename(doc.metadata.get("source", ""))
                doc.metadata["source"] = source
                docs.append(doc)

            logger.info("Loaded %d documents from %s", len(docs), self.data_path)
            return docs
        except Exception as exc:
            raise RuntimeError(f"Error processing folder: {exc}") from exc