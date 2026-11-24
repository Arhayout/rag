"""Tests for grounded answer context formatting."""

import pytest
from langchain_core.documents import Document

from src.answer_generator import build_context, generate_legal_answer


def test_context_includes_source_labels_page_and_full_text() -> None:
    documents = [
        Document(
            page_content="Short summary",
            metadata={
                "source": "/data/france/fr/code.txt",
                "language": "fr",
                "page": 2,
                "full_content": "Article 1. Full legal provision.",
            },
        )
    ]

    context = build_context(documents)

    assert "[S1]" in context
    assert "code.txt" in context
    assert "page 3" in context
    assert "Article 1. Full legal provision." in context


def test_answer_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        generate_legal_answer(
            question="What does the law say?",
            documents=[Document(page_content="Legal text", metadata={})],
            jurisdiction="FR",
            language="en",
        )


def test_context_truncates_long_sources() -> None:
    document = Document(page_content="x" * 20, metadata={"source": "law.txt"})
    assert len(build_context([document], max_chars_per_source=5).splitlines()[-1]) == 5
