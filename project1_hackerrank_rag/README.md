# HackerRank RAG Skills Insight Generator

Retrieval-Augmented Generation (RAG) system that answers questions about the HackerRank Developer Skills Report using LlamaIndex.

## Architecture

```
PDF documents (data/)
        │
        ▼
PyMuPDFReader ──► chunking ──► HuggingFace embeddings (bge-small-en-v1.5)
        │
        ▼
VectorStoreIndex (skills_report/)
        │
        ├── retrieve-only CLI (no API key)
        └── ReActAgent + OpenAI (with OPENAI_API_KEY)
```

| Component | Role |
|-----------|------|
| `src/hackerrank.py` | Index build/load, retrieval |
| `src/agent.py` | ReAct agent with query engine tool |
| `src/config.py` | Paths and model settings |
| `main.py` | CLI entry point |

## Installation

```bash
cd project1_hackerrank_rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_data.sh   # downloads HackerRank 2025 report PDF
cp .env.example .env            # optional, for agent mode
```

## Usage

**Retrieve-only (local embeddings, no API key):**

```bash
python main.py "What AI skills are most in demand?"
python main.py --retrieve-only "Which backend frameworks are popular?"
```

**Agent mode (requires `OPENAI_API_KEY` in `.env`):**

```bash
python main.py "Summarize hiring trends for junior developers"
```

**Python API:**

```python
from src.agent import get_agent

agent = get_agent()
response = agent.chat("What skills are most in demand?")
print(response)
```

## Tests

```bash
pytest -q
```

## Sample Output

```
Question: What skills are most in demand for developers?

--- Result 1 (score=0.752) ---
Developers are choosing their own learning paths...
Back-end: Spring Boot, Node.js, and .NET
Front-end: React
Data Science & ML: ML development, LLM and AI APIs...
```

## Requirements

- Python 3.10+
- ~2 GB disk for embedding model cache (first run)
- OpenAI API key (agent mode only)