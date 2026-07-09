import os

from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI

from src.hackerrank import get_query_engine

CONTEXT = """
Purpose: The primary role of this agent is to assist users by
providing accurate information about HackerRank support articles
from attached tools only.
"""

TOOL_DESCRIPTION = (
    "Use this tool to retrieve information about HackerRank support articles "
    "and skills reports. Input should be a question about HackerRank."
)

TOOL_NAME = "hackerrank_support"

_agent = None


def get_agent():
    """Returns ReActAgent."""
    global _agent
    if _agent is not None:
        return _agent

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is required for the agent. "
            "Set it in your environment or .env file."
        )

    llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
    tools = [
        QueryEngineTool(
            get_query_engine(),
            metadata=ToolMetadata(
                description=TOOL_DESCRIPTION,
                name=TOOL_NAME,
            ),
        )
    ]

    _agent = ReActAgent.from_tools(
        tools=tools,
        llm=llm,
        context=CONTEXT,
        verbose=True,
    )
    return _agent