# Multilingual Legal AI Copilot

Streamlit legal research assistant for Morocco and France. Supports English, French and Arabic UI, separate legal corpora, Groq-grounded answers, user PDF review, voice transcription, cross-jurisdiction comparison, PDF export and manual RSS/Atom checks.

## Run

```bash
cd project2_ldsa
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Configure secrets in `.streamlit/secrets.toml` (never commit this file):

```toml
GROQ_API_KEY = "your-groq-api-key"
GROQ_MODEL = "openai/gpt-oss-120b"
```

## Features

- English / Français / العربية interface; dark mode by default with a top toggle.
- Jurisdiction selector: Morocco or France. The legal indexes are loaded independently.
- Legal question and real-life situation analysis, grounded in retrieved legal documents.
- PDF contract upload and review. Uploads are parsed and indexed in memory for the current session only; scanned PDFs need OCR.
- Morocco–France comparison only runs when both jurisdictions have retrieved evidence. Each source is labelled by jurisdiction.
- Voice input uses Groq Whisper transcription; browser microphone permission is required.
- Answers can be exported to PDF.
- Manual RSS/Atom feed checks. Paste a trusted official feed URL; there is no background scheduler or push notification service.

## Legal corpora

- Morocco: `data/morocco/ar/` and `data/morocco/fr/`
- France: `data/france/fr/`

Only add verified official documents. The app never substitutes one country's law for the other. A missing corpus is reported rather than guessed.

## Privacy and limitations

- Uploaded contract files are not written to disk or added to the shared legal indexes. They remain in session memory while the app session is active.
- RSS checks are user-triggered and session-local; scheduled alerts require a separate hosting/scheduler integration.
- Scanned PDFs are not OCR-processed.
- AI answers are capped at 100 words including citations. Always verify current official law and seek qualified legal advice for decisions.
- Configure API credentials in Streamlit Secrets or environment variables. Never commit keys.

## Validation

```bash
pip install -r requirements-dev.txt
ruff check .
ruff format --check .
mypy --ignore-missing-imports --check-untyped-defs src app.py main.py scripts
pytest -q
```
