"""Unit tests for the HackerRank RAG retrieval pipeline."""

from unittest.mock import Mock

import pytest
from llama_index.core.schema import Document

import src.hackerrank as hackerrank
from src.config import PROJECT_ROOT


def test_project_layout() -> None:
    assert PROJECT_ROOT.name == "project1_hackerrank_rag"
    assert (PROJECT_ROOT / "src" / "hackerrank.py").is_file()
    assert (PROJECT_ROOT / "main.py").is_file()


def test_load_documents_requires_pdf(tmp_path) -> None:
    with pytest.raises(hackerrank.DataNotFoundError):
        hackerrank._load_documents(tmp_path)


def test_load_documents_uses_pdf_reader(monkeypatch, tmp_path) -> None:
    (tmp_path / "report.pdf").touch()
    reader = Mock()
    directory_reader = Mock()
    expected_documents = [Mock()]
    directory_reader.load_data.return_value = expected_documents

    monkeypatch.setattr(hackerrank, "PyMuPDFReader", lambda: reader)
    monkeypatch.setattr(
        hackerrank, "SimpleDirectoryReader", lambda *args, **kwargs: directory_reader
    )

    documents = hackerrank._load_documents(tmp_path)

    assert documents == expected_documents
    directory_reader.load_data.assert_called_once()


def test_get_index_builds_and_persists(monkeypatch, tmp_path) -> None:
    index = Mock()
    vector_index = Mock()
    vector_index.from_documents.return_value = index
    monkeypatch.setattr(hackerrank, "VectorStoreIndex", vector_index)
    index_path = tmp_path / "index"

    result = hackerrank.get_index([Document(text="test")], index_path)

    assert result is index
    index.storage_context.persist.assert_called_once_with(persist_dir=str(index_path))


def test_get_index_loads_existing_index(monkeypatch, tmp_path) -> None:
    index_path = tmp_path / "index"
    index_path.mkdir()
    storage_context = Mock()
    index = Mock()
    monkeypatch.setattr(
        hackerrank.StorageContext, "from_defaults", Mock(return_value=storage_context)
    )
    monkeypatch.setattr(hackerrank, "load_index_from_storage", Mock(return_value=index))

    assert hackerrank.get_index([], index_path) is index


def test_retrieve_chunks_uses_requested_top_k(monkeypatch) -> None:
    index = Mock()
    retriever = Mock()
    retriever.retrieve.return_value = ["result"]
    index.as_retriever.return_value = retriever
    monkeypatch.setattr(hackerrank, "get_index_instance", lambda: index)

    assert hackerrank.retrieve_chunks("question", top_k=2) == ["result"]
    index.as_retriever.assert_called_once_with(similarity_top_k=2)
    retriever.retrieve.assert_called_once_with("question")
