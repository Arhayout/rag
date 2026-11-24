# Moroccan Legal Document Search Assistant — Arabic + French

Bilingual legal-document retrieval for Moroccan legislation. Arabic (`ar`) and French (`fr`) documents are indexed together, so a query in either language can retrieve relevant material in either language.

## Corpus and source policy

The source registry is `data/moroccan_legal_sources.json`. It points to the Moroccan Ministry of Justice's Adala legal portal and records known version limitations. Downloaded files belong in:

```text
data/morocco/
  ar/   # Arabic PDFs or UTF-8 text
  fr/   # French PDFs or UTF-8 text
```

Priority corpus:

- Code pénal / مجموعة القانون الجنائي
- Code du travail / مدونة الشغل
- Code de la famille (Moudawana) / مدونة الأسرة
- Code des obligations et des contrats / قانون الالتزامات والعقود
- Official administrative regulations / النصوص التنظيمية والإدارية الرسمية

**Legal currency:** Arabic text published in the Official Bulletin is controlling. French translations on Adala can be significantly older (for example, the French labour-code PDF is consolidated to 2011 while the Arabic PDF is consolidated to 16 July 2026). Never treat an old French translation as current law. Keep publication/consolidation dates, law numbers, article numbers, and source URLs in document metadata. Verify amendments against the Official Bulletin before answering.

## Install and run

```bash
cd project2_ldsa
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # optional: set OPENAI_API_KEY for OpenAI embeddings/summaries
```

Place official PDFs or UTF-8 text files under `data/morocco/ar/` and `data/morocco/fr/`. The loader accepts both PDF and TXT and tags each page with its language and Moroccan jurisdiction. Image-only/scanned PDFs need OCR before indexing.

Rebuild and query:

```bash
python main.py --rebuild "ما هي شروط عقد الشغل؟"
python main.py "Quelles sont les conditions du contrat de travail ?"
streamlit run app.py
```

Use a multilingual embedding model (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) for cross-language retrieval. The model downloads on first run. Rebuild the FAISS index whenever corpus files or the embedding model change.

## Official sources

- [Ministry of Justice — Adala, French portal](https://adala.justice.gov.ma/fr)
- [Ministry of Justice — Adala, Arabic portal](https://adala.justice.gov.ma/)
- [Criminal matters (FR)](https://adala.justice.gov.ma/fr/resources/424) · [القانون الجنائي (AR)](https://adala.justice.gov.ma/resources/1053)
- [Social matters (FR)](https://adala.justice.gov.ma/fr/resources/420) · [المادة الاجتماعية (AR)](https://adala.justice.gov.ma/resources/420)
- [Civil matters (FR)](https://adala.justice.gov.ma/fr/resources/415) · [القانون المدني (AR)](https://adala.justice.gov.ma/resources/1052)
- [Family matters (AR)](https://adala.justice.gov.ma/resources/22)

## Important limitations

This is a document-search prototype, not a legal-advice system. It must cite the source document and article/page for every answer, distinguish Arabic controlling text from French translation, disclose missing or stale translations, and abstain when the retrieved corpus does not support a conclusion. Do not use generated summaries as substitutes for the legal text.
