"""OpenAI summaries with a dependency-free extractive fallback."""

from __future__ import annotations

import os
import re
from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from src.config import LLM_MODEL, SUMMARY_MAX_WORDS

SUMMARY_PROMPT = PromptTemplate(
    input_variables=["document"],
    template=(
        f"Summarize this legal document in at most {SUMMARY_MAX_WORDS} words. "
        "Capture the key legal issues, parties, and outcomes. Be factual.\n\n"
        "Document:\n{document}\n\nSummary:"
    ),
)


@lru_cache(maxsize=1)
def _openai_model() -> ChatOpenAI:
    return ChatOpenAI(model=LLM_MODEL, temperature=0)


def extractive_summary(document: str, max_words: int = SUMMARY_MAX_WORDS) -> str:
    """Return the first max_words words as a deterministic fallback."""
    words = re.sub(r"\s+", " ", document).strip().split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + "..."


def generate_summary(
    document: str, llm: ChatOpenAI | None = None
) -> str:
    """Use OpenAI when configured; otherwise summarize extractively."""
    if not os.getenv("OPENAI_API_KEY"):
        return extractive_summary(document)

    chain = SUMMARY_PROMPT | (llm or _openai_model()) | StrOutputParser()
    return chain.invoke({"document": document})
