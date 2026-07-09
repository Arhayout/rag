"""Tests for FAISS vector store operations."""

import shutil
from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.config import PROJECT_ROOT
from src.document_loader import LegalDocumentLoader
from src.vector_store import LegalDocumentStore

TEST_STORE = PROJECT_ROOT / "tests" / "_tmp_vector_store"


@pytest.fixture(autouse=True)
def cleanup_store():
    yield
    if TEST_STORE.exists():
        shutil.rmtree(TEST_STORE)


def test_save_and_query_index():
    documents = LegalDocumentLoader().load_documents()[:2]
    store = LegalDocumentStore(vecstore_path=TEST_STORE)
    vectorstore = store.save_index(documents)

    results = LegalDocumentStore.query_similar(
        vectorstore,
        "copyright infringement in streaming media",
        k=1,
    )
    assert len(results) == 1
    assert results[0].page_content


def test_query_validation():
    store = LegalDocumentStore(vecstore_path=TEST_STORE)
    docs = [Document(page_content="test legal content", metadata={"source": "t.txt"})]
    vectorstore = store.save_index(docs)

    with pytest.raises(ValueError):
        LegalDocumentStore.query_similar(vectorstore, "", k=1)
    with pytest.raises(ValueError):
        LegalDocumentStore.query_similar(vectorstore, "query", k=0)