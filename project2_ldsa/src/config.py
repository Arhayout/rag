"""Application configuration for the Legal Document Search Assistant."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"

EMBEDDING_MODEL_OPENAI = "text-embedding-ada-002"
EMBEDDING_MODEL_LOCAL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "gpt-4o-mini"
SUMMARY_MAX_WORDS = 100
DEFAULT_TOP_K = 5