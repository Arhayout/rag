import argparse
import os
import sys

from dotenv import load_dotenv

from src.config import DEFAULT_QUESTION
from src.hackerrank import retrieve_chunks


def retrieve_only(question: str) -> None:
    nodes = retrieve_chunks(question)
    print(f"Question: {question}\n")
    for index, node in enumerate(nodes, start=1):
        score = f"{node.score:.3f}" if node.score is not None else "n/a"
        print(f"--- Result {index} (score={score}) ---")
        print(node.text[:1000].strip())
        print()


def run_agent(question: str) -> None:
    from src.agent import get_agent

    print(get_agent().chat(question))


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="HackerRank RAG skills insight generator"
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUESTION,
        help="Question about the HackerRank skills report",
    )
    parser.add_argument(
        "--retrieve-only",
        action="store_true",
        help="Use retrieval without the ReAct agent",
    )
    args = parser.parse_args()

    try:
        if args.retrieve_only or not os.getenv("OPENAI_API_KEY"):
            if not os.getenv("OPENAI_API_KEY"):
                print("OPENAI_API_KEY not set. Running retrieval-only mode.\n")
            retrieve_only(args.question)
        else:
            run_agent(args.question)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
