# Moroccan Legal Search

Bilingual Arabic/French semantic search over Moroccan legal documents. The app uses LangChain, multilingual sentence-transformer embeddings, and FAISS.

## Local setup

Use Python 3.12.

```bash
cd project2_ldsa
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On first launch, the app downloads available PDFs registered in `data/moroccan_legal_sources.json` from the Moroccan Ministry of Justice Adala portal. To download them manually, run `python scripts/download_moroccan_sources.py`.

Review every source, consolidation date, and amendment before relying on it. The Arabic Official Bulletin is authoritative; a French translation may be older or incomplete.

OpenAI is optional. Without `OPENAI_API_KEY`, the project uses local multilingual embeddings and extractive summaries. If configured, OpenAI is used for document summaries.

## Deploy on Streamlit Community Cloud

1. Open https://share.streamlit.io/ and sign in with GitHub.
2. Select repository `Arhayout/rag`, branch `main`, and app file `project2_ldsa/app.py`.
3. Select Python 3.12.
4. Add `OPENAI_API_KEY` under app Secrets only if you want OpenAI summaries.
5. Deploy.

The app downloads the registered legal PDFs on first startup; no API key is required for local embeddings or retrieval. Streamlit Community Cloud has resource limits, so first startup can take several minutes while the embedding model is downloaded and the index is built.

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
