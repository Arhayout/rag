"""Tests for HackerRank RAG retrieval pipeline."""

from pathlib import Path

import pytest

from src.config import DATA_DIR, PROJECT_ROOT


def test_project_layout():
    assert PROJECT_ROOT.name == "project1_hackerrank_rag"
    assert (PROJECT_ROOT / "src" / "hackerrank.py").exists()
    assert (PROJECT_ROOT / "main.py").exists()


def test_data_directory_exists():
    assert DATA_DIR.exists()


@pytest.mark.skipif(
    not list(DATA_DIR.glob("*.pdf")),
    reason="PDF data not present; run scripts/download_data.sh",
)
def test_retrieve_chunks_returns_results():
    from src.hackerrank import retrieve_chunks

    nodes = retrieve_chunks("What backend skills are in demand?", top_k=2)
    assert len(nodes) == 2
    assert all(node.text.strip() for node in nodes)
    assert all(node.score is not None for node in nodes)