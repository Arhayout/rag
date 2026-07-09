"""Application configuration for the HackerRank RAG project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = PROJECT_ROOT / "skills_report"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_TOP_K = 4
DEFAULT_QUESTION = "What skills are most in demand for developers?"
LLM_MODEL = "gpt-4o-mini"