# Multilingual Legal Research Assistant

A Streamlit legal RAG application with an English/French/Arabic interface, separate Moroccan and French corpora, multilingual semantic retrieval, and optional answer generation through Groq.

## Local setup

Use Python 3.12.

```bash
cd project2_ldsa
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Jurisdictions and corpus layout

The indexes and source directories are isolated by jurisdiction:

- Morocco: `data/morocco/ar/` and `data/morocco/fr/`
- France: `data/france/fr/`

Place verified PDF or UTF-8 TXT legal documents in the matching language folder. The app never searches one jurisdiction's index while another is selected. Morocco retains its official-source downloader. The France corpus is intentionally empty until French legal documents are supplied; the app does not silently substitute Moroccan law.

For Morocco, the Arabic Official Bulletin is authoritative. French translations can be older or incomplete. For France, prioritise current official consolidated texts and check amendments and effective dates before relying on a result.

## Groq answer generation

Retrieval works locally without a Groq key. To generate an answer from retrieved passages, add a Groq API key to Streamlit Secrets:

```toml
GROQ_API_KEY = "your-groq-api-key"
GROQ_MODEL = "openai/gpt-oss-120b"
```

Alternatively, set `GROQ_API_KEY` and optionally `GROQ_MODEL` as environment variables. Never commit secrets to Git. The default model is Groq's production `openai/gpt-oss-120b`; it can be overridden in Secrets.

The answer prompt restricts claims to retrieved passages and requires source markers such as `[S1]`. The UI displays the corresponding document names and page numbers. If the retrieved material is insufficient, the model is instructed to say so. AI output can still be wrong; verify the cited official text.

## Validation

```bash
pip install -r requirements-dev.txt
ruff check .
ruff format --check .
mypy --ignore-missing-imports --check-untyped-defs src app.py main.py scripts
pytest -q
```

## Legal notice

This is a document retrieval and information tool, not legal advice. Verify the original official text, amendments, consolidation date, and effective dates.
