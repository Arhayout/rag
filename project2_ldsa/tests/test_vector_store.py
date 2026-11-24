"""Tests for FAISS index persistence and retrieval."""

from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings

from src.vector_store import LegalDocumentStore


@pytest.fixture
def fake_store_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.vector_store.get_embeddings", lambda: FakeEmbeddings(size=32)
    )
    monkeypatch.setattr(
        "src.vector_store.generate_summary", lambda text: text[:100]
    )


def test_save_load_and_query_index(
    tmp_path: Path, fake_store_dependencies: None
) -> None:
    documents = [
        Document(page_content="Moroccan labour law", metadata={"source": "labour.txt"}),
        Document(page_content="Moroccan commercial law", metadata={"source": "trade.txt"}),
    ]
    store = LegalDocumentStore(tmp_path)
    vectorstore = store.save_index(documents)

    loaded = LegalDocumentStore.load_local(tmp_path)
    assert loaded is not None
    results = LegalDocumentStore.query_similar(loaded, "labour", k=1)

    assert len(results) == 1
    assert results[0].metadata["source"] in {"labour.txt", "trade.txt"}


def test_save_index_rejects_empty_documents(
    tmp_path: Path, fake_store_dependencies: None
) -> None:
    with pytest.raises(ValueError, match="empty document list"):
        LegalDocumentStore(tmp_path).save_index([])


def test_query_validation(
    tmp_path: Path, fake_store_dependencies: None
) -> None:
    vectorstore = LegalDocumentStore(tmp_path).save_index(
        [Document(page_content="legal content", metadata={"source": "law.txt"})]
    )

    with pytest.raises(ValueError, match="Query must not be empty"):
        LegalDocumentStore.query_similar(vectorstore, " ", k=1)
    with pytest.raises(ValueError, match="k must be at least 1"):
        LegalDocumentStore.query_similar(vectorstore, "query", k=0)
