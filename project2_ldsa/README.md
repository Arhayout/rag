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

No API key is required. The app uses local multilingual embeddings by default. If `OPENAI_API_KEY` is configured, OpenAI is used for document summaries. OpenAI embeddings are opt-in via `USE_OPENAI_EMBEDDINGS=true` and incur API charges.

## Deploy on Streamlit Community Cloud

1. Open https://share.streamlit.io/ and sign in with GitHub.
2. Select repository `Arhayout/rag`, branch `main`, and app file `project2_ldsa/app.py`.
3. Select Python 3.12.
4. Leave Secrets empty for the free local-embedding mode. Add `OPENAI_API_KEY` only if you want OpenAI summaries.
5. Deploy.

The app downloads the registered legal PDFs on first startup; no API key is required for retrieval. Streamlit Community Cloud has resource limits, so first startup can take several minutes while the embedding model is downloaded and the index is built.

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
