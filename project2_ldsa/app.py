"""Streamlit interface for Moroccan legal document search."""

from __future__ import annotations

import logging

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

from src.config import DATA_DIR, DEFAULT_TOP_K, VECTOR_STORE_DIR
from src.document_loader import LegalDocumentLoader
from src.vector_store import LegalDocumentStore

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TEXT = {
    "Français": {
        "title": "Recherche juridique marocaine",
        "caption": "Recherche sémantique dans les textes juridiques marocains en arabe et en français.",
        "question": "Votre question juridique",
        "placeholder": "Ex. : durée légale du travail au Maroc",
        "top_k": "Nombre de résultats",
        "search": "Rechercher",
        "loading": "Recherche dans les textes...",
        "empty": "Aucun résultat pertinent.",
        "source": "Source",
        "language": "Langue",
        "disclaimer": "Outil documentaire, pas un conseil juridique. Vérifiez toujours le texte officiel et ses modifications.",
        "missing": "Corpus juridique absent ou vide.",
        "setup": "Téléchargez les sources officielles depuis le dossier du projet, puis redémarrez l'application :",
        "failure": "Impossible d'initialiser la recherche. Consultez les logs de l'application.",
    },
    "العربية": {
        "title": "البحث في التشريع المغربي",
        "caption": "بحث دلالي في النصوص القانونية المغربية بالعربية والفرنسية.",
        "question": "السؤال القانوني",
        "placeholder": "مثال: مدة العمل القانونية في المغرب",
        "top_k": "عدد النتائج",
        "search": "بحث",
        "loading": "جارٍ البحث في النصوص...",
        "empty": "لم يتم العثور على نتائج مناسبة.",
        "source": "المصدر",
        "language": "اللغة",
        "disclaimer": "أداة للبحث الوثائقي وليست استشارة قانونية. يرجى التحقق من النص الرسمي وتعديلاته.",
        "missing": "المجموعة القانونية غير موجودة أو فارغة.",
        "setup": "حمّل المصادر الرسمية من مجلد المشروع ثم أعد تشغيل التطبيق:",
        "failure": "تعذر تشغيل البحث. راجع سجلات التطبيق.",
    },
}


@st.cache_resource(show_spinner="Initialisation du moteur de recherche…")
def initialize_retrieval_system() -> FAISS:
    """Load the saved index or build it from the local legal corpus."""
    vectorstore = LegalDocumentStore.load_local(VECTOR_STORE_DIR)
    if vectorstore is not None:
        return vectorstore

    documents = LegalDocumentLoader(DATA_DIR).load_documents()
    return LegalDocumentStore(VECTOR_STORE_DIR).save_index(documents)


def render_search_results(
    vectorstore: FAISS, query: str, top_k: int, labels: dict[str, str]
) -> None:
    """Display matching passages with their source metadata."""
    results = LegalDocumentStore.query_similar(vectorstore, query, top_k)
    if not results:
        st.info(labels["empty"])
        return

    for index, document in enumerate(results, start=1):
        metadata = document.metadata
        source = metadata.get("source", "unknown")
        language = metadata.get("language", "—")
        page = metadata.get("page")
        details = f"{labels['source']}: {source} · {labels['language']}: {language}"
        if page is not None:
            details += f" · page {page + 1}"

        with st.expander(f"{index}. {details}"):
            st.markdown(document.page_content)
            full_content = metadata.get("full_content")
            if full_content:
                with st.expander("Texte intégral / النص الكامل"):
                    st.write(full_content)


def main() -> None:
    st.set_page_config(page_title="Moroccan Legal Search", page_icon="⚖️")
    language = st.selectbox("Interface / الواجهة", tuple(TEXT))
    labels = TEXT[language]

    st.title(labels["title"])
    st.caption(labels["caption"])
    st.info(labels["disclaimer"])

    try:
        vectorstore = initialize_retrieval_system()
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("Legal corpus is unavailable: %s", exc)
        st.warning(labels["missing"])
        st.write(labels["setup"])
        st.code("python scripts/download_moroccan_sources.py")
        return
    except Exception:  # noqa: BLE001 - show a safe message at the UI boundary
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
                render_search_results(vectorstore, query, int(top_k), labels)
            except Exception:  # noqa: BLE001 - isolate errors from user input
                logger.exception("Legal search failed")
                st.error(labels["failure"])


if __name__ == "__main__":
    main()
