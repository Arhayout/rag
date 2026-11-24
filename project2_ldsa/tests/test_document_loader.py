"""Tests for Moroccan Arabic/French legal document loading."""

from pathlib import Path

from src.document_loader import LegalDocumentLoader


def test_load_documents_returns_arabic_and_french_metadata(tmp_path: Path) -> None:
    for language, filename, body in (
        ("ar", "labour.txt", "نص قانوني باللغة العربية"),
        ("fr", "travail.txt", "Texte juridique en français"),
    ):
        folder = tmp_path / language
        folder.mkdir()
        (folder / filename).write_text(body, encoding="utf-8")

    documents = LegalDocumentLoader(tmp_path).load_documents()

    assert len(documents) == 2
    assert {doc.metadata["language"] for doc in documents} == {"ar", "fr"}
    assert all(doc.metadata["jurisdiction"] == "MA" for doc in documents)
    assert all(doc.metadata["document_type"] == "moroccan_legislation" for doc in documents)
    assert all(Path(doc.metadata["source"]).is_file() for doc in documents)
