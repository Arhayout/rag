# Moroccan Legal Search

Bilingual Arabic/French semantic search over Moroccan legal documents. The app uses LangChain, multilingual sentence-transformer embeddings, and FAISS.

## Local setup

Use Python 3.12.

```bash
cd project2_ldsa
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_moroccan_sources.py
streamlit run app.py
```

The downloader saves available registered PDFs to `data/morocco/ar/` and `data/morocco/fr/`. Review every source, consolidation date, and amendment before relying on it. The Arabic Official Bulletin is authoritative; a French translation may be older or incomplete.

OpenAI is optional. Without `OPENAI_API_KEY`, the project uses local multilingual embeddings and extractive summaries. If configured, OpenAI is used for document summaries.

## Deploy on Streamlit Community Cloud

1. Open https://share.streamlit.io/ and sign in with GitHub.
2. Select repository `Arhayout/rag`, branch `main`, and app file `project2_ldsa/app.py`.
3. Select Python 3.12.
4. Add `OPENAI_API_KEY` under app Secrets only if you want OpenAI summaries.
5. Deploy.

The Streamlit app uses `project2_ldsa/requirements.txt`. Add the legal PDFs to the deployment environment or run the downloader in the project before deployment. The repository does not include the source PDFs.

## Validation

```bash
pip install -r requirements-dev.txt
ruff check .
ruff format --check .
mypy --ignore-missing-imports --check-untyped-defs src app.py main.py scripts
pytest -q
```

## Legal notice

This is a document retrieval aid, not legal advice. Search results are not a statement of current law. Verify the original official text, amendments, and effective dates before use.
