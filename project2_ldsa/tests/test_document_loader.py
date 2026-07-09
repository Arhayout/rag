"""Tests for legal document loading."""

from src.config import DATA_DIR
from src.document_loader import LegalDocumentLoader


def test_load_documents_returns_txt_files():
    loader = LegalDocumentLoader(DATA_DIR)
    docs = loader.load_documents()
    assert len(docs) >= 5
    assert all(doc.page_content.strip() for doc in docs)
    assert all(doc.metadata.get("source", "").endswith(".txt") for doc in docs)