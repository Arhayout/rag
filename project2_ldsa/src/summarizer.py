"""Document summarization with OpenAI or extractive fallback."""

from __future__ import annotations

import os
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from src.config import LLM_MODEL, SUMMARY_MAX_WORDS


SUMMARY_PROMPT = PromptTemplate(
    input_variables=["document"],
    template=(
        "Summarize the following legal document in no more than "
        f"{SUMMARY_MAX_WORDS} words. Capture key legal issues, parties, "
        "and outcomes. Be concise and factual.\n\n"
        "Document:\n{document}\n\nSummary:"
    ),
)


def extractive_summary(document: str, max_words: int = SUMMARY_MAX_WORDS) -> str:
    """Fallback summary using the first sentences up to max_words."""
    text = re.sub(r"\s+", " ", document).strip()
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def generate_summary(document: str, llm: ChatOpenAI | None = None) -> str:
    """Generate a summary using OpenAI when available, else extractive."""
    if os.getenv("OPENAI_API_KEY"):
        model = llm or ChatOpenAI(model=LLM_MODEL, temperature=0)
        chain = SUMMARY_PROMPT | model | StrOutputParser()
        return chain.invoke({"document": document})

    return extractive_summary(document)