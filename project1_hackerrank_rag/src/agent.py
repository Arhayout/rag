import os

from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI

from src.config import LLM_MODEL
from src.hackerrank import get_query_engine

CONTEXT = """Purpose: Answer questions using only the attached HackerRank support
articles and skills reports."""


TOOL_DESCRIPTION = (
    "Retrieve information about HackerRank support articles and skills reports. "
    "Input should be a question about HackerRank."
)
TOOL_NAME = "hackerrank_support"

_agent: ReActAgent | None = None


def get_agent() -> ReActAgent:
    """Return the cached ReAct agent, creating it on first use."""
    global _agent
    if _agent is not None:
        return _agent

    if not os.getenv("OPENAI_API_KEY"):
        raise OSError(
            "OPENAI_API_KEY is required for the agent. "
            "Set it in your environment or .env file."
        )

    llm = OpenAI(model=LLM_MODEL, temperature=0.1)
    tool = QueryEngineTool(
        query_engine=get_query_engine(),
        metadata=ToolMetadata(description=TOOL_DESCRIPTION, name=TOOL_NAME),
    )
    _agent = ReActAgent.from_tools(
        tools=[tool],
        llm=llm,
        context=CONTEXT,
        verbose=True,
    )
    return _agent
