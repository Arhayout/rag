"""Tests for legal document loading."""

from src.config import DATA_DIR
from src.document_loader import LegalDocumentLoader


def test_load_documents_returns_txt_files() -> None:
    documents = LegalDocumentLoader(DATA_DIR).load_documents()

    assert len(documents) >= 5
    assert all(document.page_content.strip() for document in documents)
    assert all(
        document.metadata.get("source", "").endswith(".txt")
        for document in documents
    )
