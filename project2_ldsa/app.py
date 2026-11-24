"""Multilingual legal copilot for French and Moroccan law."""
from __future__ import annotations

import io
import logging
import os
from xml.sax.saxutils import escape

import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from reportlab.lib.units import cm

from src.assistant_features import answer_with_groq, build_upload_index, search_index, transcribe_audio, read_legal_feed
from src.config import DEFAULT_TOP_K, JURISDICTIONS, GROQ_MODEL
from src.document_loader import LegalDocumentLoader
from src.vector_store import LegalDocumentStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GROQ_MODELS = {
    "GPT-OSS 120B · Advanced reasoning": "openai/gpt-oss-120b",
    "Qwen 3.8 27B · Multilingual (preview)": "qwen/qwen3.8-27b",
    "GPT-OSS 20B · Faster reasoning": "openai/gpt-oss-20b",
}

UI = {
    "English": {
        "title":"Legal AI Copilot","subtitle":"Source-grounded legal research for Morocco and France.","country":"Legal jurisdiction",
        "dark":"☀️ Light mode","light":"🌙 Dark mode","question":"Ask a legal question","placeholder":"Describe your legal question or situation…",
        "ask":"Analyze","mode":"Analysis type","general":"Legal question","situation":"Legal situation","voice":"🎙️ Ask by voice",
        "transcribe":"Transcribe audio","answer":"AI answer","upload":"Upload contract or PDF","contract":"Analyze document",
        "contract_prompt":"What should I check in this document?","compare":"Compare French and Moroccan law",
        "compare_q":"What legal issue should be compared?","updates":"Legal updates","feed":"Official RSS feed URL",
        "check":"Check updates","no_feed":"Manual checks only. HTTPS feeds must use an allowlisted official government domain.",
        "empty":"No relevant official sources found. Add verified legal documents first.","disclaimer":"Legal information only, not legal advice. Verify current official texts and consult a qualified lawyer.",
        "download":"Download answer as PDF","no_key":"Configure GROQ_API_KEY in Streamlit Secrets.","upload_help":"PDF only, maximum 15 MB per file. Processed in memory for this session.",
        "law_context":"Also retrieve official legal sources","notification":"Updates found","language":"Interface","model":"AI model",
    },
    "Français": {
        "title":"Assistant juridique IA","subtitle":"Recherche juridique sourcée pour le Maroc et la France.","country":"Juridiction",
        "dark":"☀️ Mode clair","light":"🌙 Mode sombre","question":"Poser une question juridique","placeholder":"Décrivez votre question ou situation juridique…",
        "ask":"Analyser","mode":"Type d’analyse","general":"Question juridique","situation":"Situation concrète","voice":"🎙️ Question vocale",
        "transcribe":"Transcrire l’audio","answer":"Réponse IA","upload":"Importer un contrat ou PDF","contract":"Analyser le document",
        "contract_prompt":"Que souhaitez-vous vérifier dans ce document ?","compare":"Comparer le droit français et marocain",
        "compare_q":"Quel sujet juridique comparer ?","updates":"Veille juridique","feed":"URL du flux RSS officiel",
        "check":"Vérifier les nouveautés","no_feed":"Vérification manuelle uniquement. Flux HTTPS hébergés sur un domaine gouvernemental officiel autorisé.",
        "empty":"Aucune source officielle pertinente trouvée. Ajoutez d’abord des textes juridiques vérifiés.","disclaimer":"Information juridique, pas un conseil juridique. Vérifiez les textes officiels en vigueur et consultez un avocat.",
        "download":"Télécharger la réponse en PDF","no_key":"Configurez GROQ_API_KEY dans les Secrets Streamlit.","upload_help":"PDF uniquement, 15 Mo maximum par fichier. Traitement en mémoire pendant la session.",
        "law_context":"Rechercher aussi les sources juridiques officielles","notification":"Nouveautés trouvées","language":"Interface","model":"Modèle IA",
    },
    "العربية": {
        "title":"المساعد القانوني الذكي","subtitle":"بحث قانوني موثق لقوانين المغرب وفرنسا.","country":"النظام القانوني",
        "dark":"☀️ الوضع الفاتح","light":"🌙 الوضع الداكن","question":"اطرح سؤالاً قانونياً","placeholder":"اشرح سؤالك أو وضعك القانوني…",
        "ask":"تحليل","mode":"نوع التحليل","general":"سؤال قانوني","situation":"حالة قانونية","voice":"🎙️ سؤال صوتي",
        "transcribe":"تحويل الصوت إلى نص","answer":"إجابة الذكاء الاصطناعي","upload":"تحميل عقد أو ملف PDF","contract":"تحليل المستند",
        "contract_prompt":"ما الذي تريد التحقق منه في هذا المستند؟","compare":"مقارنة القانون المغربي والفرنسي",
        "compare_q":"ما الموضوع القانوني المطلوب مقارنته؟","updates":"التحديثات القانونية","feed":"رابط موجز RSS رسمي",
        "check":"تحقق من التحديثات","no_feed":"التحقق يدوي فقط. يجب أن يكون رابط HTTPS على نطاق حكومي رسمي معتمد.",
        "empty":"لم يتم العثور على مصادر رسمية مناسبة. أضف نصوصاً قانونية موثقة أولاً.","disclaimer":"معلومات قانونية وليست استشارة قانونية. تحقق من النصوص الرسمية واستشر محامياً مختصاً.",
        "download":"تحميل الإجابة PDF","no_key":"أضف GROQ_API_KEY إلى Streamlit Secrets.","upload_help":"ملفات PDF فقط، بحد أقصى 15 ميغابايت لكل ملف. تتم المعالجة في الذاكرة.",
        "law_context":"البحث أيضاً في المصادر القانونية الرسمية","notification":"تحديثات جديدة","language":"لغة الواجهة","model":"نموذج الذكاء الاصطناعي",
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
    story = [Paragraph(escape(title), styles["Title"]), Spacer(1, 12)]
    for paragraph in body.splitlines():
        if paragraph.strip():
            story.extend([Paragraph(escape(paragraph), styles["BodyText"]), Spacer(1, 7)])
    doc.build(story)
    return buffer.getvalue()


def show_answer(answer: str, title: str, sources: list) -> None:
    st.subheader(title)
    st.markdown(answer)
    if sources:
        st.caption("Sources")
        for i, doc in enumerate(sources, 1):
            meta = doc.metadata
            source = str(meta.get("source", "Uploaded document"))
            page = meta.get("page")
            st.markdown(f"**[S{i}]** {source}" + (f" — page {int(page)+1}" if page is not None else ""))
            with st.expander(f"Excerpt [S{i}]"):
                st.write(doc.page_content[:1000])
    st.download_button(UI[st.session_state.ui_language]["download"], data=make_pdf(title, answer),
                       file_name="legal-answer.pdf", mime="application/pdf")


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
        language = st.selectbox("Interface / Langue / اللغة", list(UI), index=list(UI).index(st.session_state.ui_language))
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

    country = st.selectbox(t["country"], ["Morocco", "France"])
    model_labels = list(GROQ_MODELS)
    default_model = os.environ.get("GROQ_MODEL", GROQ_MODEL)
    default_model_index = list(GROQ_MODELS.values()).index(default_model) if default_model in GROQ_MODELS.values() else 0
    selected_model_label = st.selectbox(t["model"], model_labels, index=default_model_index)
    selected_model = GROQ_MODELS[selected_model_label]
    code = "MA" if country == "Morocco" else "FR"
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
                transcript = transcribe_audio(audio.getvalue(), output_language)
                st.session_state.voice_transcript = transcript
                st.success(transcript)
            except Exception:
                logger.exception("Audio transcription failed")
                st.error(t["no_key"])
        if st.session_state.get("voice_transcript"):
            query = st.text_area("Transcript", value=st.session_state.voice_transcript, key="voice_text")
        if st.button(t["ask"], type="primary") and query.strip():
            if index is None:
                st.warning(t["empty"])
            else:
                docs = search_index(index, query, DEFAULT_TOP_K)
                if docs:
                    task = ("Analyse this real-life legal situation. Identify issues, applicable rules, risks, missing facts and practical next steps."
                            if mode == t["situation"] else "Answer the legal question concisely.")
                    try:
                        answer = answer_with_groq(query, docs, code, output_language, task, model=selected_model)
                        st.session_state.latest_answer, st.session_state.latest_sources = answer, docs
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
                    "Review the uploaded contract. Separate explicit text from legal interpretation. Flag ambiguous clauses, obligations, deadlines, risks and questions for a lawyer. Do not claim illegality unless retrieved law supports it.",
                    extra_context="Uploaded files: " + st.session_state.upload_name,
                    model=selected_model,
                )
                st.session_state.latest_answer, st.session_state.latest_sources = answer, combined
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
                    answer = answer_with_groq(
                        comparison_query, ma_docs + fr_docs, "Morocco and France", output_language,
                        "Compare the two legal systems. Label every proposition Morocco or France. Never merge rules.",
                        extra_context="MOROCCO SOURCES:\n" + "\n".join(d.page_content[:700] for d in ma_docs)
                                     + "\nFRANCE SOURCES:\n" + "\n".join(d.page_content[:700] for d in fr_docs),
                        model=selected_model,
                    )
                    st.session_state.latest_answer, st.session_state.latest_sources = answer, ma_docs + fr_docs
                    show_answer(answer, t["compare"], ma_docs + fr_docs)
                except RuntimeError:
                    st.error(t["no_key"])

    with tabs[3]:
        st.info(t["no_feed"])
        feed_url = st.text_input(t["feed"], placeholder="https://…/feed.xml")
        if st.button(t["check"]) and feed_url:
            try:
                st.session_state.feed_items = read_legal_feed(feed_url)
            except Exception as exc:
                st.error(str(exc))
        for item in st.session_state.feed_items:
            st.markdown(f"- **{item['title']}** — {item['published']}  \n  {item['link']}")
        if st.session_state.feed_items:
            st.success(t["notification"])


if __name__ == "__main__":
    main()
