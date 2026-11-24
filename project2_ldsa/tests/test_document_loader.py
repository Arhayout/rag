"""Tests for jurisdiction-specific legal document loading."""

from pathlib import Path

import pytest

from src.document_loader import LegalDocumentLoader


def test_load_documents_adds_moroccan_language_and_jurisdiction_metadata(
    tmp_path: Path,
) -> None:
    samples = {
        "ar": "نص قانوني باللغة العربية",
        "fr": "Texte juridique en français",
    }
    for language, content in samples.items():
        folder = tmp_path / language
        folder.mkdir()
        (folder / "law.txt").write_text(content, encoding="utf-8")

    documents = LegalDocumentLoader(tmp_path, jurisdiction="MA").load_documents()

    assert len(documents) == 2
    assert {document.metadata["language"] for document in documents} == {"ar", "fr"}
    assert all(document.metadata["jurisdiction"] == "MA" for document in documents)
    assert all(
        document.metadata["document_type"] == "moroccan_legislation"
        for document in documents
    )
    assert all(Path(document.metadata["source"]).is_file() for document in documents)


def test_loader_supports_french_jurisdiction_without_mixing_moroccan_files(
    tmp_path: Path,
) -> None:
    (tmp_path / "fr").mkdir()
    (tmp_path / "ar").mkdir()
    (tmp_path / "fr" / "code.txt").write_text(
        "French legislation", encoding="utf-8"
    )
    (tmp_path / "ar" / "code.txt").write_text(
        "Moroccan Arabic legislation", encoding="utf-8"
    )

    documents = LegalDocumentLoader(tmp_path, jurisdiction="FR").load_documents()

    assert len(documents) == 1
    assert documents[0].page_content == "French legislation"
    assert documents[0].metadata["jurisdiction"] == "FR"
    assert documents[0].metadata["document_type"] == "french_legislation"


def test_loader_rejects_empty_corpus(tmp_path: Path) -> None:
    (tmp_path / "ar").mkdir()
    with pytest.raises(ValueError, match="No PDF or TXT documents"):
        LegalDocumentLoader(tmp_path).load_documents()


def test_loader_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Corpus directory"):
        LegalDocumentLoader(tmp_path / "missing")


def test_loader_rejects_unknown_jurisdiction(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported jurisdiction"):
        LegalDocumentLoader(tmp_path, jurisdiction="US")
