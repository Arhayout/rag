"""Embedding provider factory with local embeddings as the default."""

from __future__ import annotations

import os

from langchain_core.embeddings import Embeddings

from src.config import EMBEDDING_MODEL_LOCAL


def get_embeddings() -> Embeddings:
    """Use local multilingual embeddings unless paid OpenAI embeddings are enabled."""
    use_openai = os.getenv("USE_OPENAI_EMBEDDINGS", "").lower() in {
        "1",
        "true",
        "yes",
    }
    if use_openai:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "USE_OPENAI_EMBEDDINGS requires an OPENAI_API_KEY."
            )
        from langchain_openai import OpenAIEmbeddings

        from src.config import EMBEDDING_MODEL_OPENAI

        return OpenAIEmbeddings(model=EMBEDDING_MODEL_OPENAI)

    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_LOCAL)
