import argparse
import os
import sys

from dotenv import load_dotenv

from src.config import DEFAULT_QUESTION
from src.hackerrank import get_query_engine, retrieve_chunks


def retrieve_only(question: str) -> None:
    nodes = retrieve_chunks(question)
    print(f"Question: {question}\n")
    for i, node in enumerate(nodes, 1):
        score = f"{node.score:.3f}" if node.score is not None else "n/a"
        print(f"--- Result {i} (score={score}) ---")
        print(node.text[:1000].strip())
        print()


def run_agent(question: str) -> None:
    from src.agent import get_agent

    agent = get_agent()
    response = agent.chat(question)
    print(response)


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="HackerRank RAG Skills Insight Generator")
    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUESTION,
        help="Question to ask about the HackerRank skills report",
    )
    parser.add_argument(
        "--retrieve-only",
        action="store_true",
        help="Use retrieval + synthesis without the ReAct agent",
    )
    args = parser.parse_args()

    try:
        if args.retrieve_only or not os.getenv("OPENAI_API_KEY"):
            if not os.getenv("OPENAI_API_KEY"):
                print("OPENAI_API_KEY not set. Running retrieve-only mode.\n")
            retrieve_only(args.question)
        else:
            run_agent(args.question)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()