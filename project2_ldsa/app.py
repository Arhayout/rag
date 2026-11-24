"""Multilingual legal copilot for French and Moroccan law.

Answers are capped at 100 words. Uploaded files and RSS checks remain session-local.
"""
from __future__ import annotations

import io
import logging
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from reportlab.lib.units import cm

from src.assistant_features import (
    answer_with_groq, build_upload_index, search_index, transcribe_audio,
    read_legal_feed, trim_words,
)
from src.config import DEFAULT_TOP_K, JURISDICTIONS
from src.document_loader import LegalDocumentLoader
from src.vector_store import LegalDocumentStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UI = {
    "English": {
        "title": "Legal AI Copilot", "subtitle": "Source-grounded legal research for Morocco and France.",
        "language": "Interface", "country": "Legal jurisdiction", "dark": "☀️ Light mode",
        "light": "🌙 Dark mode", "question": "Ask a legal question",
        "placeholder": "Describe your legal question or situation…", "ask": "Analyze",
        "mode": "Analysis type", "general": "Legal question", "situation": "Legal situation",
        "voice": "🎙️ Ask by voice", "transcribe": "Transcribe audio", "answer": "AI answer",
        "sources": "Official legal sources", "upload": "Upload contract or PDF",
        "contract": "Analyze document", "contract_prompt": "What should I check in this document?",
        "compare": "Compare French and Moroccan law", "compare_q": "What legal issue should be compared?",
        "updates": "Legal updates", "feed": "Official RSS feed URL", "check": "Check updates",
        "no_feed": "Enter an official RSS/Atom feed URL. Checks are manual and session-only.",
        "empty": "No relevant official sources found in this jurisdiction. Add verified legal documents first.",
        "disclaimer": "Legal information only, not legal advice. Verify current official texts and consult a qualified lawyer.",
        "download": "Download answer as PDF", "no_key": "Configure GROQ_API_KEY in Streamlit Secrets.",
        "upload_help": "PDF files only. Files are processed in memory and are not added to the public legal corpus.",
        "law_context": "Also retrieve official legal sources", "notification": "Updates found",
    },
    "Français": {
        "title": "Assistant juridique IA", "subtitle": "Recherche juridique sourcée pour le Maroc et la France.",
        "language": "Interface", "country": "Juridiction", "dark": "☀️ Mode clair",
        "light": "🌙 Mode sombre", "question": "Poser une question juridique",
        "placeholder": "Décrivez votre question ou situation juridique…", "ask": "Analyser",
        "mode": "Type d’analyse", "general": "Question juridique", "situation": "Situation concrète",
        "voice": "🎙️ Question vocale", "transcribe": "Transcrire l’audio", "answer": "Réponse IA",
        "sources": "Sources juridiques officielles", "upload": "Importer un contrat ou PDF",
        "contract": "Analyser le document", "contract_prompt": "Que souhaitez-vous vérifier dans ce document ?",
        "compare": "Comparer le droit français et marocain", "compare_q": "Quel sujet juridique comparer ?",
        "updates": "Veille juridique", "feed": "URL du flux RSS officiel", "check": "Vérifier les nouveautés",
        "no_feed": "Saisissez l’URL d’un flux RSS/Atom officiel. Vérification manuelle, limitée à la session.",
        "empty": "Aucune source officielle pertinente trouvée. Ajoutez d’abord des textes juridiques vérifiés.",
        "disclaimer": "Information juridique, pas un conseil juridique. Vérifiez les textes officiels en vigueur et consultez un avocat.",
        "download": "Télécharger la réponse en PDF", "no_key": "Configurez GROQ_API_KEY dans les Secrets Streamlit.",
        "upload_help": "PDF uniquement. Les fichiers sont traités en mémoire et ne rejoignent pas le corpus public.",
        "law_context": "Rechercher aussi les sources juridiques officielles", "notification": "Nouveautés trouvées",
    },
    "العربية": {
        "title": "المساعد القانوني الذكي", "subtitle": "بحث قانوني موثق لقوانين المغرب وفرنسا.",
        "language": "لغة الواجهة", "country": "النظام القانوني", "dark": "☀️ الوضع الفاتح",
        "light": "🌙 الوضع الداكن", "question": "اطرح سؤالاً قانونياً",
        "placeholder": "اشرح سؤالك أو وضعك القانوني…", "ask": "تحليل",
        "mode": "نوع التحليل", "general": "سؤال قانوني", "situation": "حالة قانونية",
        "voice": "🎙️ سؤال صوتي", "transcribe": "تحويل الصوت إلى نص", "answer": "إجابة الذكاء الاصطناعي",
        "sources": "المصادر القانونية الرسمية", "upload": "تحميل عقد أو ملف PDF",
        "contract": "تحليل المستند", "contract_prompt": "ما الذي تريد التحقق منه في هذا المستند؟",
        "compare": "مقارنة القانون المغربي والفرنسي", "compare_q": "ما الموضوع القانوني المطلوب مقارنته؟",
        "updates": "التحديثات القانونية", "feed": "رابط موجز RSS رسمي", "check": "تحقق من التحديثات",
        "no_feed": "أدخل رابط موجز RSS/Atom رسمي. التحقق يدوي ومؤقت خلال الجلسة.",
        "empty": "لم يتم العثور على مصادر رسمية مناسبة. أضف نصوصاً قانونية موثقة أولاً.",
        "disclaimer": "معلومات قانونية وليست استشارة قانونية. تحقق من النصوص الرسمية واستشر محامياً مختصاً.",
        "download": "تحميل الإجابة PDF", "no_key": "أضف GROQ_API_KEY إلى Streamlit Secrets.",
        "upload_help": "ملفات PDF فقط. تتم المعالجة في الذاكرة ولا تضاف إلى corpus العام.",
        "law_context": "البحث أيضاً في المصادر القانونية الرسمية", "notification": "تحديثات جديدة",
    },
}


@st.cache_resource(show_spinner=False)
def get_legal_index(code: str):
    config = JURISDICTIONS[code]
    store = LegalDocumentStore(config["vector_store_dir"])
    index = store.load_local(config["vector_store_dir"])
    if index is not None:
        return index
    try:
        docs = LegalDocumentLoader(data_path=config["data_dir"], jurisdiction=code).load_documents()
        return store.save_index(docs)
    except (FileNotFoundError, ValueError):
        return None


def make_pdf(title: str, body: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 12)]
    for paragraph in body.splitlines():
        if paragraph.strip():
            story.extend([Paragraph(paragraph.replace("&", "&amp;"), styles["BodyText"]), Spacer(1, 7)])
    doc.build(story)
    return buffer.getvalue()


def show_answer(answer: str, title: str, sources: list) -> None:
    st.subheader(title)
    st.markdown(answer)
    if sources:
        st.caption("Sources")
        for i, doc in enumerate(sources, 1):
            meta = doc.metadata
            source = meta.get("source", "Uploaded document")
            page = meta.get("page")
            st.markdown(f"**[S{i}]** {source}" + (f" — page {int(page)+1}" if page is not None else ""))
            with st.expander(f"Excerpt [S{i}]"):
                st.write(doc.page_content[:1200])
    st.download_button(
        UI[st.session_state.ui_language]["download"],
        data=make_pdf(title, answer),
        file_name="legal-answer.pdf",
        mime="application/pdf",
    )


def main() -> None:
    st.set_page_config(page_title="Legal AI Copilot", page_icon="⚖️", layout="wide")
    st.session_state.setdefault("theme", "dark")
    st.session_state.setdefault("ui_language", "Français")
    st.session_state.setdefault("upload_index", None)
    st.session_state.setdefault("upload_name", "")
    st.session_state.setdefault("latest_answer", "")
    st.session_state.setdefault("feed_items", [])

    header, theme_col = st.columns([5, 1])
    with header:
        language = st.selectbox("Interface", list(UI), index=list(UI).index(st.session_state.ui_language), label_visibility="collapsed")
        st.session_state.ui_language = language
    t = UI[language]
    with theme_col:
        if st.button(t["dark"] if st.session_state.theme == "dark" else t["light"], use_container_width=True):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()

    dark = st.session_state.theme == "dark"
    bg, fg, panel = ("#171a21", "#e6e9ef", "#222833") if dark else ("#f7f8fa", "#20242b", "#ffffff")
    st.markdown(f"<style>.stApp{{background:{bg};color:{fg}}} [data-testid='stSidebar']{{background:{panel}}}</style>", unsafe_allow_html=True)

    st.title(t["title"])
    st.caption(t["subtitle"])
    jurisdiction_label = st.selectbox(t["country"], ["Morocco", "France"], format_func=lambda x: "المغرب" if language == "العربية" and x == "Morocco" else ("فرنسا" if language == "العربية" else x))
    code = "MA" if jurisdiction_label == "Morocco" else "FR"
    output_language = {"English": "en", "Français": "fr", "العربية": "ar"}[language]
    index = get_legal_index(code)
    if index is None:
        st.warning(t["empty"])

    tabs = st.tabs([t["question"], t["upload"], t["compare"], t["updates"]])

    with tabs[0]:
        mode = st.radio(t["mode"], [t["general"], t["situation"]], horizontal=True)
        query = st.text_area(t["question"], placeholder=t["placeholder"], max_chars=5000)
        audio = st.audio_input(t["voice"])
        if audio and st.button(t["transcribe"]):
            try:
                transcript = transcribe_audio(audio.getvalue(), language=output_language)
                st.session_state.voice_transcript = transcript
                st.success(transcript)
            except Exception:
                logger.exception("Audio transcription failed")
                st.error("Audio transcription unavailable. Check Groq configuration.")
        if st.session_state.get("voice_transcript"):
            query = st.text_area("Transcript", value=st.session_state.voice_transcript, key="voice_text")
        if st.button(t["ask"], type="primary") and query.strip():
            if not index:
                st.warning(t["empty"])
            else:
                docs = search_index(index, query, DEFAULT_TOP_K)
                if docs:
                    task = "Analyse this real-life legal situation. Identify issues, applicable rules, risks, missing facts and practical next steps." if mode == t["situation"] else "Answer the legal question concisely."
                    try:
                        answer = answer_with_groq(query, docs, code, output_language, task)
                        st.session_state.latest_answer = answer
                        st.session_state.latest_sources = docs
                    except RuntimeError:
                        st.error(t["no_key"])
                else:
                    st.info(t["empty"])
        if st.session_state.latest_answer:
            show_answer(st.session_state.latest_answer, t["answer"], st.session_state.get("latest_sources", []))
        st.caption(t["disclaimer"])

    with tabs[1]:
        uploads = st.file_uploader(t["upload"], type=["pdf"], accept_multiple_files=True, help=t["upload_help"])
        if uploads and st.button(t["upload"]):
            try:
                st.session_state.upload_index = build_upload_index(uploads)
                st.session_state.upload_name = ", ".join(file.name for file in uploads)
                st.success(st.session_state.upload_name)
            except Exception as exc:
                st.error(f"Could not read PDF: {exc}")
        contract_question = st.text_area(t["contract_prompt"])
        include_law = st.checkbox(t["law_context"], value=True)
        if st.button(t["contract"]) and st.session_state.upload_index and contract_question.strip():
            contract_docs = search_index(st.session_state.upload_index, contract_question, 6)
            legal_docs = search_index(index, contract_question, 4) if include_law and index else []
            combined = contract_docs + legal_docs
            try:
                answer = answer_with_groq(
                    contract_question, combined, code, output_language,
                    "Review the uploaded contract. Separate what the document explicitly says from legal interpretation. Flag ambiguous clauses, obligations, deadlines, risks and questions for a lawyer. Do not claim a clause is illegal unless retrieved law supports it.",
                    extra_context="Uploaded files: " + st.session_state.upload_name,
                )
                st.session_state.latest_answer = answer
                st.session_state.latest_sources = combined
                show_answer(answer, t["contract"], combined)
            except RuntimeError:
                st.error(t["no_key"])
        st.caption(t["disclaimer"])

    with tabs[2]:
        comparison_query = st.text_area(t["compare_q"])
        if st.button(t["compare"]) and comparison_query.strip():
            indexes = {c: get_legal_index(c) for c in ("MA", "FR")}
            ma_docs = search_index(indexes["MA"], comparison_query, 4) if indexes["MA"] else []
            fr_docs = search_index(indexes["FR"], comparison_query, 4) if indexes["FR"] else []
            if not ma_docs or not fr_docs:
                st.warning("Comparison requires verified sources in both jurisdictions. One corpus is currently empty.")
            else:
                try:
                    comparison = answer_with_groq(
                        comparison_query, ma_docs + fr_docs, "COMPARISON", output_language,
                        "Compare the two legal systems in a structured way. Label every proposition Morocco or France. Do not merge rules. Explain differences, similarities and source gaps.",
                        extra_context="MOROCCO SOURCES:\n" + "\n".join(d.page_content[:900] for d in ma_docs) + "\nFRANCE SOURCES:\n" + "\n".join(d.page_content[:900] for d in fr_docs),
                    )
                    st.session_state.latest_answer = comparison
                    st.session_state.latest_sources = ma_docs + fr_docs
                    show_answer(comparison, t["compare"], ma_docs + fr_docs)
                except RuntimeError:
                    st.error(t["no_key"])

    with tabs[3]:
        st.info(t["no_feed"])
        feed_url = st.text_input(t["feed"], placeholder="https://…/feed.xml")
        if st.button(t["check"]) and feed_url.startswith("https://"):
            try:
                st.session_state.feed_items = read_legal_feed(feed_url)
            except Exception:
                st.error("Feed unavailable. Verify that this is a trusted official RSS/Atom URL.")
        for item in st.session_state.feed_items:
            st.markdown(f"- **{item['title']}** — {item['published']}  \n  {item['link']}")
        if st.session_state.feed_items:
            st.caption(t["notification"])

if __name__ == "__main__":
    main()
