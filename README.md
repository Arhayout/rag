# RAG Portfolio Projects

Two production-style Retrieval-Augmented Generation (RAG) applications demonstrating different stacks and use cases.

## Projects

| Project | Stack | Use Case |
|---------|-------|----------|
| [project1_hackerrank_rag](./project1_hackerrank_rag) | LlamaIndex + HuggingFace + OpenAI agent | Query HackerRank Developer Skills Report PDFs |
| [project2_ldsa](./project2_ldsa) | LangChain + FAISS + Streamlit | Semantic search over legal case documents |

## Quick Start

```bash
# Project 1
cd project1_hackerrank_rag
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_data.sh
python main.py --retrieve-only

# Project 2
cd ../project2_ldsa
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py "Copyright disputes in digital media"
```

## Design Principles

- Modular `src/` packages with dedicated config modules
- Local-first execution (no API key required for demos)
- Optional OpenAI integration for higher-quality synthesis
- Pytest coverage for core pipelines
- Persistent vector indexes for fast repeat queries

## License

MIT