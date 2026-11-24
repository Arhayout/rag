"""Application configuration for the Moroccan bilingual legal RAG."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "morocco"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store_morocco"

EMBEDDING_MODEL_OPENAI = "text-embedding-3-small"
# Multilingual model: supports Arabic and French cross-lingual retrieval.
EMBEDDING_MODEL_LOCAL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "gpt-4o-mini"
SUMMARY_MAX_WORDS = 180
DEFAULT_TOP_K = 5
