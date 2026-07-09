"""Embedding provider factory."""

from __future__ import annotations

import os

from langchain_core.embeddings import Embeddings


def get_embeddings() -> Embeddings:
    """Return OpenAI embeddings when configured, otherwise local HuggingFace."""
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import OpenAIEmbeddings

        from src.config import EMBEDDING_MODEL_OPENAI

        return OpenAIEmbeddings(model=EMBEDDING_MODEL_OPENAI)

    from src.config import EMBEDDING_MODEL_LOCAL

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_LOCAL)