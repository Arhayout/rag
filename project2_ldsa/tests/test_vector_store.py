"""Tests for FAISS vector store operations."""

import shutil

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings

from src.config import PROJECT_ROOT
from src.document_loader import LegalDocumentLoader
from src.vector_store import LegalDocumentStore

TEST_STORE = PROJECT_ROOT / "tests" / "_tmp_vector_store"


@pytest.fixture(autouse=True)
def use_fake_embeddings(monkeypatch):
    monkeypatch.setattr(
        "src.vector_store.get_embeddings", lambda: FakeEmbeddings(size=32)
    )
    yield
    if TEST_STORE.exists():
        shutil.rmtree(TEST_STORE)


def test_save_and_query_index() -> None:
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


def test_query_validation() -> None:
    documents = [Document(page_content="test legal content", metadata={"source": "t.txt"})]
    vectorstore = LegalDocumentStore(vecstore_path=TEST_STORE).save_index(documents)

    with pytest.raises(ValueError, match="Query must not be empty"):
        LegalDocumentStore.query_similar(vectorstore, "", k=1)
    with pytest.raises(ValueError, match="k must be at least 1"):
        LegalDocumentStore.query_similar(vectorstore, "query", k=0)
