"""Command-line interface for Moroccan legal document search."""

from __future__ import annotations

import argparse
import logging
import sys

from dotenv import load_dotenv

from src.config import DEFAULT_TOP_K, VECTOR_STORE_DIR
from src.document_loader import LegalDocumentLoader
from src.vector_store import LegalDocumentStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def search(query: str, top_k: int) -> None:
    vectorstore = LegalDocumentStore.load_local(VECTOR_STORE_DIR)
    if vectorstore is None:
        documents = LegalDocumentLoader().load_documents()
        vectorstore = LegalDocumentStore().save_index(documents)

    results = LegalDocumentStore.query_similar(vectorstore, query, top_k)
    print(f"Query: {query}\n")
    for index, document in enumerate(results, start=1):
        source = document.metadata.get("source", "unknown")
        print(f"--- Result {index} [{source}] ---")
        print(document.page_content)
        print()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Moroccan legal document search")
    parser.add_argument("query", help="Legal question or topic")
    parser.add_argument("-k", "--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--rebuild", action="store_true", help="Rebuild the FAISS index"
    )
    args = parser.parse_args()

    try:
        if args.rebuild:
            documents = LegalDocumentLoader().load_documents()
            LegalDocumentStore().save_index(documents)
        search(args.query, args.top_k)
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
