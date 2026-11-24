"""Configuration for the multilingual, multi-jurisdiction legal RAG."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

JURISDICTIONS = {
    "MA": {
        "name": "Morocco",
        "data_dir": DATA_ROOT / "morocco",
        "vector_store_dir": PROJECT_ROOT / "vector_store_morocco",
        "languages": ("ar", "fr"),
        "document_type": "moroccan_legislation",
    },
    "FR": {
        "name": "France",
        "data_dir": DATA_ROOT / "france",
        "vector_store_dir": PROJECT_ROOT / "vector_store_france",
        "languages": ("fr",),
        "document_type": "french_legislation",
    },
}

DATA_DIR = JURISDICTIONS["MA"]["data_dir"]
VECTOR_STORE_DIR = JURISDICTIONS["MA"]["vector_store_dir"]
EMBEDDING_MODEL_OPENAI = "text-embedding-3-small"
EMBEDDING_MODEL_LOCAL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "gpt-4o-mini"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "openai/gpt-oss-120b"
SUMMARY_MAX_WORDS = 100
DEFAULT_TOP_K = 5
