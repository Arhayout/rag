"""Streamlit interface for multilingual legal search and grounded answers."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

from scripts.download_moroccan_sources import download_sources
from src.answer_generator import generate_legal_answer
from src.config import DEFAULT_TOP_K, JURISDICTIONS
from src.document_loader import LegalDocumentLoader
from src.vector_store import LegalDocumentStore

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TEXT = {
    "English": {
        "title": "Legal research assistant",
        "caption": "Search legal documents and generate source-grounded answers.",
        "interface": "Interface language",
        "jurisdiction": "Jurisdiction",
        "morocco": "Morocco",
        "france": "France",
        "question": "Your legal question",
        "placeholder": "e.g. What is the statutory working time?",
        "top_k": "Number of sources",
        "search": "Search and answer",
        "loading": "Searching legal sources…",
        "answer_loading": "Generating an answer with Groq…",
        "answer": "AI-generated answer",
        "sources": "Retrieved sources",
        "source": "Source",
        "language": "Language",
        "empty": "No relevant passages found.",
        "missing": "This jurisdiction's legal corpus is empty.",
        "setup_fr": "Add verified French legislation under data/france/fr/. The French corpus is kept separate from Moroccan sources.",
        "setup_ma": "The Moroccan corpus could not be downloaded. Check the source registry and Adala links.",
        "no_key": "Search is available, but answer generation requires GROQ_API_KEY in Streamlit Secrets or the environment.",
        "failure": "The search or answer service could not be completed. Check application logs.",
        "disclaimer": "Document retrieval is not legal advice. Verify the current official text, amendments, and effective dates.",
    },
    "Français": {
        "title": "Assistant de recherche juridique",
        "caption": "Rechercher dans les textes et générer des réponses fondées sur les sources.",
        "interface": "Langue de l’interface",
        "jurisdiction": "Juridiction",
        "morocco": "Maroc",
        "france": "France",
        "question": "Votre question juridique",
        "placeholder": "Ex. : Quelle est la durée légale du travail ?",
        "top_k": "Nombre de sources",
        "search": "Rechercher et répondre",
        "loading": "Recherche dans les textes…",
        "answer_loading": "Génération de la réponse avec Groq…",
        "answer": "Réponse générée par l’IA",
        "sources": "Sources retrouvées",
        "source": "Source",
        "language": "Langue",
        "empty": "Aucun passage pertinent trouvé.",
        "missing": "Le corpus juridique de cette juridiction est vide.",
        "setup_fr": "Ajoutez les textes français vérifiés dans data/france/fr/. Le corpus français reste séparé des sources marocaines.",
        "setup_ma": "Le corpus marocain n’a pas pu être téléchargé. Vérifiez le registre des sources et les liens Adala.",
        "no_key": "La recherche fonctionne, mais la génération nécessite GROQ_API_KEY dans les Secrets Streamlit ou l’environnement.",
        "failure": "Le service de recherche ou de réponse a échoué. Consultez les logs.",
        "disclaimer": "La recherche documentaire ne constitue pas un conseil juridique. Vérifiez le texte officiel en vigueur, ses modifications et ses dates d’effet.",
    },
    "العربية": {
        "title": "مساعد البحث القانوني",
        "caption": "البحث في النصوص القانونية وإنشاء إجابات مستندة إلى المصادر.",
        "interface": "لغة الواجهة",
        "jurisdiction": "الاختصاص القضائي",
        "morocco": "المغرب",
        "france": "فرنسا",
        "question": "سؤالك القانوني",
        "placeholder": "مثال: ما هي مدة العمل القانونية؟",
        "top_k": "عدد المصادر",
        "search": "بحث وإجابة",
        "loading": "جارٍ البحث في النصوص…",
        "answer_loading": "جارٍ إنشاء الإجابة عبر Groq…",
        "answer": "الإجابة المُنشأة بالذكاء الاصطناعي",
        "sources": "المصادر المسترجعة",
        "source": "المصدر",
        "language": "اللغة",
        "empty": "لم يتم العثور على مقاطع ذات صلة.",
        "missing": "المجموعة القانونية لهذه الولاية القضائية فارغة.",
        "setup_fr": "أضف النصوص الفرنسية الموثوقة إلى data/france/fr/. تبقى المجموعة الفرنسية منفصلة عن المصادر المغربية.",
        "setup_ma": "تعذر تنزيل المجموعة المغربية. تحقق من سجل المصادر وروابط عدالة.",
        "no_key": "البحث متاح، لكن إنشاء الإجابات يتطلب GROQ_API_KEY في Streamlit Secrets أو متغيرات البيئة.",
        "failure": "تعذر إكمال البحث أو إنشاء الإجابة. راجع سجلات التطبيق.",
        "disclaimer": "استرجاع الوثائق ليس استشارة قانونية. تحقق من النص الرسمي الساري وتعديلاته وتواريخ نفاذه.",
    },
}


def get_secret(name: str) -> str | None:
    """Read a secret without requiring it to exist in local development."""
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets.get(name)
    except (AttributeError, FileNotFoundError):
        return None


@st.cache_resource(show_spinner=False)
def initialize_retrieval_system(jurisdiction: str) -> FAISS:
    """Load or build an isolated vector index for one jurisdiction."""
    config = JURISDICTIONS[jurisdiction]
    data_dir = Path(config["data_dir"])
    vector_dir = Path(config["vector_store_dir"])

    vectorstore = LegalDocumentStore.load_local(vector_dir)
    if vectorstore is not None:
        return vectorstore

    has_documents = data_dir.is_dir() and any(
        path.is_file() and path.suffix.lower() in {".pdf", ".txt"}
        for path in data_dir.rglob("*")
    )
    if not has_documents and jurisdiction == "MA":
        download_sources()

    documents = LegalDocumentLoader(
        data_path=data_dir,
        jurisdiction=jurisdiction,
    ).load_documents()
    return LegalDocumentStore(vector_dir).save_index(documents)


def render_sources(
    documents: list,
    labels: dict[str, str],
) -> None:
    """Display retrieved passages and their source metadata."""
    if not documents:
        st.info(labels["empty"])
        return

    st.subheader(labels["sources"])
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata
        source = Path(str(metadata.get("source", "unknown"))).name
        language = metadata.get("language", "—")
        page = metadata.get("page")
        details = f"[S{index}] {labels['source']}: {source} · {labels['language']}: {language}"
        if page is not None:
            details += f" · p. {int(page) + 1}"
        with st.expander(details):
            st.markdown(document.page_content)
            full_content = metadata.get("full_content")
            if full_content:
                with st.expander("Full text / Texte intégral / النص الكامل"):
                    st.write(full_content)


def apply_theme() -> None:
    """Apply a restrained, high-readability theme."""
    dark = st.session_state.get("theme", "dark") == "dark"
    background = "#171a21" if dark else "#f7f8fa"
    foreground = "#e6e9ef" if dark else "#20242b"
    panel = "#222833" if dark else "#ffffff"
    border = "#394252" if dark else "#d8dde6"
    muted = "#aeb7c4" if dark else "#626b78"
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stToolbar"] {{ background: {background}; color: {foreground}; }}
        [data-testid="stSidebar"] {{ background: {panel}; color: {foreground}; border-right: 1px solid {border}; }}
        [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li, label, h1, h2, h3, h4, [data-testid="stWidgetLabel"] p {{ color: {foreground}; }}
        [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input, [data-testid="stSelectbox"] [role="combobox"] {{ background: {panel}; color: {foreground}; border: 1px solid {border}; border-radius: 8px; }}
        [data-testid="stExpander"], [data-testid="stAlert"] {{ background: {panel}; border: 1px solid {border}; color: {foreground}; border-radius: 10px; }}
        [data-testid="stCaptionContainer"], .stCaption {{ color: {muted}; }}
        [data-testid="stButton"] button {{ border: 1px solid {border}; border-radius: 8px; color: {foreground}; background: {panel}; }}
        [data-testid="stButton"] button:hover {{ border-color: #7c9bd1; color: {foreground}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Legal Research Assistant", page_icon="⚖️")
    st.session_state.setdefault("theme", "dark")
    language_column, theme_column = st.columns([3, 1.5], vertical_alignment="center")
    with language_column:
        language = st.selectbox(
            "Interface language / Langue / اللغة",
            options=tuple(TEXT),
            format_func=lambda value: value,
            label_visibility="collapsed",
        )
    with theme_column:
        button_label = (
            "☀️ Light mode"
            if st.session_state["theme"] == "dark"
            else "🌙 Dark mode"
        )
        if st.button(button_label, use_container_width=True):
            st.session_state["theme"] = (
                "light" if st.session_state["theme"] == "dark" else "dark"
            )
            st.rerun()
    apply_theme()

    labels = TEXT[language]

    st.title(labels["title"])
    st.caption(labels["caption"])
    st.info(labels["disclaimer"])

    jurisdiction = st.selectbox(
        labels["jurisdiction"],
        options=("MA", "FR"),
        format_func=lambda value: (
            labels["morocco"] if value == "MA" else labels["france"]
        ),
    )

    try:
        vectorstore = initialize_retrieval_system(jurisdiction)
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("Legal corpus is unavailable for %s: %s", jurisdiction, exc)
        st.warning(labels["missing"])
        st.write(labels["setup_fr"] if jurisdiction == "FR" else labels["setup_ma"])
        return
    except Exception:  # noqa: BLE001 - safe message at the UI boundary
        logger.exception("Could not initialize legal retrieval")
        st.error(labels["failure"])
        return

    query = st.text_input(labels["question"], placeholder=labels["placeholder"])
    top_k = st.number_input(
        labels["top_k"], min_value=1, max_value=10, value=DEFAULT_TOP_K
    )

    if st.button(labels["search"], type="primary") and query.strip():
        with st.spinner(labels["loading"]):
            try:
                results = LegalDocumentStore.query_similar(
                    vectorstore, query, int(top_k)
                )
                if not results:
                    st.info(labels["empty"])
                    return
                st.subheader(labels["answer"])
                api_key = get_secret("GROQ_API_KEY")
                if not api_key:
                    st.info(labels["no_key"])
                else:
                    with st.spinner(labels["answer_loading"]):
                        answer = generate_legal_answer(
                            question=query,
                            documents=results,
                            jurisdiction=jurisdiction,
                            language={"English": "en", "Français": "fr", "العربية": "ar"}[
                                language
                            ],
                            api_key=api_key,
                            model=get_secret("GROQ_MODEL"),
                        )
                    st.markdown(answer)
                render_sources(results, labels)
            except Exception:  # noqa: BLE001 - isolate errors from user input
                logger.exception("Legal search or answer generation failed")
                st.error(labels["failure"])


if __name__ == "__main__":
    main()
