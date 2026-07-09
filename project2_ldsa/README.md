# Legal Document Search Assistant (LDSA)

Semantic search over legal case documents using LangChain, FAISS, and optional OpenAI summarization.

## Architecture

```
Legal .txt files (data/)
        │
        ▼
LegalDocumentLoader
        │
        ▼
Summarizer (OpenAI or extractive fallback)
        │
        ▼
FAISS vector index (vector_store/)
        │
        ├── CLI (main.py)
        └── Streamlit UI (app.py)
```

| Module | Responsibility |
|--------|----------------|
| `src/document_loader.py` | Load `.txt` case files |
| `src/summarizer.py` | LLM or extractive summaries |
| `src/embeddings.py` | OpenAI or local HuggingFace embeddings |
| `src/vector_store.py` | FAISS index build/load/search |
| `main.py` | CLI search |
| `app.py` | Streamlit interface |

## Installation

```bash
cd project2_ldsa
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional, enables OpenAI summaries/embeddings
```

## Usage

**CLI (works without API key — local embeddings + extractive summaries):**

```bash
python main.py "Liability in autonomous vehicle accidents"
python main.py "Patent infringement in AI technology" -k 3
python main.py --rebuild "Copyright disputes in digital media"
```

**Streamlit UI:**

```bash
streamlit run app.py --server.port 8000
```

Example queries:
- Copyright disputes in digital media
- Liability in autonomous vehicle accidents
- Patent infringement in AI technology

## Tests

```bash
pytest -q
```

## Sample Output

```
Query: Copyright disputes in digital media

--- Result 1 [copyright_digital_media.txt] ---
Summary: Case: Rivera v. StreamCast Media (2024)...
Excerpt: Independent filmmaker Elena Rivera alleged...
```

## Requirements

- Python 3.10+
- OpenAI API key optional (improves summary quality)