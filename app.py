import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_core.vectorstores import InMemoryVectorStore


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Document Q&A",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# --------------------------------------------------
# STYLING
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #0B0F17;
    --surface: #151B27;
    --surface-hover: #1B2333;
    --ink: #E8ECF4;
    --muted: #9AA6BA;
    --line: #263044;
    --accent: #3F6EF5;
    --accent-dark: #2F58D1;
    --accent-text: #8FB0FF;
    --accent-soft: #1B2745;
}

html, body, .stApp, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, 'Segoe UI', sans-serif;
}

/* ---------- Base ---------- */
.stApp {
    background: var(--bg);
    color: var(--ink);
}

[data-testid="stSidebar"],
[data-testid="collapsedControl"],
[data-testid="stToolbar"],
#MainMenu,
footer {
    display: none !important;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 760px;
    padding-top: 2.5rem;
    padding-bottom: 7rem;
}

/* ---------- Upload screen ---------- */
.hero {
    text-align: center;
    margin: 2.5rem 0 2rem 0;
}

.hero-badge {
    display: inline-block;
    background: var(--accent-soft);
    color: var(--accent-text);
    font-size: 13px;
    font-weight: 600;
    padding: 6px 14px;
    border-radius: 999px;
    margin-bottom: 18px;
}

.hero h1 {
    font-size: 42px;
    line-height: 1.15;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--ink);
    margin: 0 0 14px 0;
    padding: 0;
}

.hero p {
    font-size: 17px;
    line-height: 1.6;
    color: var(--muted);
    max-width: 520px;
    margin: 0 auto;
}

[data-testid="stFileUploader"] {
    margin-top: 1.5rem;
}

[data-testid="stFileUploaderDropzone"] {
    background: var(--surface);
    border: 2px dashed #34467A;
    border-radius: 18px;
    padding: 2.4rem 1.5rem;
    transition: border-color .15s ease, background .15s ease;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent);
    background: var(--surface-hover);
}

[data-testid="stFileUploaderDropzone"] * {
    color: var(--muted) !important;
}

[data-testid="stFileUploaderDropzone"] button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.3rem !important;
}

[data-testid="stFileUploaderDropzone"] button:hover {
    background: var(--accent-dark) !important;
}

[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploaderDropzone"] button * {
    color: #ffffff !important;
}

.trust-row {
    display: flex;
    justify-content: center;
    gap: 28px;
    flex-wrap: wrap;
    margin-top: 28px;
    color: var(--muted);
    font-size: 14px;
}

.trust-row span::before {
    content: "✓";
    color: var(--accent-text);
    font-weight: 700;
    margin-right: 8px;
}

/* ---------- Chat header ---------- */
.doc-chip {
    display: flex;
    align-items: center;
    gap: 12px;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 12px 16px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
}

.doc-chip .icon {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    background: var(--accent-soft);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
}

.doc-chip .meta {
    min-width: 0;
}

.doc-chip .label {
    font-size: 12px;
    color: var(--muted);
}

.doc-chip .name {
    font-size: 15px;
    font-weight: 600;
    color: var(--ink);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* ---------- Buttons ---------- */
.stButton > button {
    background: var(--surface);
    color: var(--ink);
    border: 1px solid var(--line);
    border-radius: 12px;
    font-weight: 600;
    font-size: 14px;
    padding: 0.55rem 1rem;
    transition: border-color .15s ease, color .15s ease, background .15s ease;
}

.stButton > button:hover {
    border-color: var(--accent);
    color: var(--accent-text);
    background: var(--surface-hover);
}

.stButton > button:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
}

.suggest-label {
    font-size: 13px;
    color: var(--muted);
    margin: 18px 0 8px 2px;
}

/* ---------- Chat messages ---------- */
[data-testid="stChatMessage"] {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 14px 18px;
    margin-bottom: 14px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
    color: var(--ink) !important;
    font-size: 15px;
    line-height: 1.65;
}

/* User messages: right-aligned, accent colored */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: var(--accent);
    border-color: var(--accent);
    margin-left: 18%;
    flex-direction: row-reverse;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p {
    color: #ffffff !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageAvatarUser"] {
    display: none;
}

/* Assistant avatar */
[data-testid="stChatMessageAvatarAssistant"] {
    background: var(--accent-soft) !important;
    color: var(--accent-text) !important;
}

/* ---------- Sources expander ---------- */
[data-testid="stExpander"] {
    border: 1px solid var(--line);
    border-radius: 12px;
    background: var(--surface);
    margin-top: 10px;
}

[data-testid="stExpander"] summary {
    font-size: 13px;
    font-weight: 600;
    color: var(--muted);
}

.source-item {
    font-size: 13px;
    color: var(--muted);
    border-left: 3px solid var(--accent-soft);
    padding-left: 12px;
    margin: 10px 0;
    line-height: 1.55;
}

.source-item b {
    color: var(--ink);
}

/* ---------- Chat input ---------- */
[data-testid="stBottom"],
[data-testid="stBottom"] > div {
    background: linear-gradient(to top, var(--bg) 70%, rgba(11, 15, 23, 0));
}

[data-testid="stChatInput"] {
    max-width: 760px;
    margin-left: auto;
    margin-right: auto;
    border-radius: 16px;
    border: 1px solid var(--line);
    background: var(--surface);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
}

[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-soft), 0 4px 16px rgba(0, 0, 0, 0.4);
}

[data-testid="stChatInput"] textarea::placeholder {
    color: var(--muted) !important;
}

[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--ink) !important;
    caret-color: var(--accent-text);
    font-size: 15px !important;
}

[data-testid="stChatInput"] button {
    background: var(--accent) !important;
    border-radius: 10px !important;
}

[data-testid="stChatInput"] button svg {
    fill: #ffffff !important;
}

/* ---------- Misc dark fixes ---------- */
.stMarkdown, [data-testid="stSpinner"] {
    color: var(--ink);
}

[data-testid="stExpander"] details {
    background: var(--surface);
}

/* ---------- Alerts ---------- */
[data-testid="stAlert"] {
    border-radius: 12px;
}

@media (max-width: 640px) {
    .hero h1 { font-size: 32px; }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        margin-left: 6%;
    }
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "upload"

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "document_name" not in st.session_state:
    st.session_state.document_name = ""

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# --------------------------------------------------
# API KEY AND MODELS
# --------------------------------------------------

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Google API key not found. Please configure GOOGLE_API_KEY.")
    st.stop()


@st.cache_resource
def load_models(api_key):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        api_key=api_key
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=api_key,
        max_output_tokens=1024
    )

    return embeddings, llm


embeddings, llm = load_models(GOOGLE_API_KEY)


# --------------------------------------------------
# PDF PROCESSING
# --------------------------------------------------

def process_pdf(uploaded_file):

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        loader = PyPDFLoader(temp_path)
        pages = loader.load()

        if not pages:
            raise ValueError("No readable text found in this PDF.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(pages)

        if not chunks:
            raise ValueError("Could not extract text from this PDF.")

        vector_db = InMemoryVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings
        )

        return vector_db

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def render_message(message):
    """Draw one chat message, with its sources if it has any."""
    role = message["role"]

    with st.chat_message(role):
        st.markdown(message["content"])

        sources = message.get("sources")
        if sources:
            with st.expander(f"Sources ({len(sources)})"):
                for source in sources:
                    st.markdown(
                        f'<div class="source-item">'
                        f'<b>Page {source["page"]}</b><br>{source["snippet"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )


def answer_question(user_question):
    """Retrieve context, ask the model, and return (answer, sources)."""
    retriever = st.session_state.vector_db.as_retriever(
        search_kwargs={"k": 4}
    )

    relevant_docs = retriever.invoke(user_question)

    context = "\n\n".join(doc.page_content for doc in relevant_docs)

    history = ""

    for message in st.session_state.messages[-7:-1]:
        role = "User" if message["role"] == "user" else "Assistant"
        history += f"{role}: {message['content']}\n"

    prompt = f"""
You are a helpful PDF question-answering assistant.

Answer the user's question using the PDF context below.

Rules:
1. Give clear and accurate answers.
2. Use the PDF context as your primary source.
3. Do not invent information.
4. If the answer is not available in the PDF, clearly say so.
5. Use simple formatting and bullet points when useful.

PDF CONTEXT:
{context}

PREVIOUS CONVERSATION:
{history}

USER QUESTION:
{user_question}

ANSWER:
"""

    response = llm.invoke(prompt)
    answer = response.content

    if isinstance(answer, list):
        answer = "\n".join(
            item.get("text", "")
            for item in answer
            if isinstance(item, dict) and item.get("type") == "text"
        )

    sources = []
    seen_pages = set()

    for doc in relevant_docs:
        page_number = doc.metadata.get("page")
        page_label = (page_number + 1) if isinstance(page_number, int) else "-"

        if page_label in seen_pages:
            continue
        seen_pages.add(page_label)

        snippet = " ".join(doc.page_content.split())
        if len(snippet) > 220:
            snippet = snippet[:220].rstrip() + "…"

        sources.append({"page": page_label, "snippet": snippet})

    return answer, sources


def reset_app():
    st.session_state.page = "upload"
    st.session_state.vector_db = None
    st.session_state.document_name = ""
    st.session_state.messages = []
    st.session_state.pending_question = None


SUGGESTED_QUESTIONS = [
    "Summarize this document",
    "What are the key points?",
    "List any important dates or numbers",
]


# --------------------------------------------------
# SCREEN 1: PDF UPLOAD
# --------------------------------------------------

if st.session_state.page == "upload":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-badge">📄 Document Q&A</div>
            <h1>Ask your PDF anything</h1>
            <p>
                Upload a document and get clear answers in seconds,
                with the pages they came from.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"],
        label_visibility="collapsed"
    )

    st.markdown(
        """
        <div class="trust-row">
            <span>Answers come from your document</span>
            <span>Page sources included</span>
            <span>Ask follow-up questions</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    if uploaded_file is not None:

        with st.spinner("Reading your document…"):

            try:
                vector_db = process_pdf(uploaded_file)

                st.session_state.vector_db = vector_db
                st.session_state.document_name = uploaded_file.name

                st.session_state.messages = [
                    {
                        "role": "assistant",
                        "content": (
                            f"Your document **{uploaded_file.name}** is ready.\n\n"
                            "Ask me a question below, or pick one of the "
                            "suggestions to get started."
                        )
                    }
                ]

                st.session_state.page = "chat"

                st.rerun()

            except Exception as error:
                st.error(f"Couldn't process this document: {error}")


# --------------------------------------------------
# SCREEN 2: CHATBOT
# --------------------------------------------------

elif st.session_state.page == "chat":

    # Top bar: document name + button to start over
    top_left, top_right = st.columns([5, 2], vertical_alignment="center")

    with top_left:
        st.markdown(
            f"""
            <div class="doc-chip">
                <div class="icon">📄</div>
                <div class="meta">
                    <div class="label">Current document</div>
                    <div class="name">{st.session_state.document_name}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with top_right:
        if st.button("Upload new PDF", use_container_width=True):
            reset_app()
            st.rerun()

    st.write("")

    # Previous conversation
    for message in st.session_state.messages:
        render_message(message)

    # Suggested questions, shown only before the first question
    if len(st.session_state.messages) == 1:
        st.markdown(
            '<div class="suggest-label">Try asking</div>',
            unsafe_allow_html=True
        )

        columns = st.columns(len(SUGGESTED_QUESTIONS))

        for column, suggestion in zip(columns, SUGGESTED_QUESTIONS):
            with column:
                if st.button(
                    suggestion,
                    key=f"suggest_{suggestion}",
                    use_container_width=True
                ):
                    st.session_state.pending_question = suggestion
                    st.rerun()

    # Chat input (or a clicked suggestion)
    typed_question = st.chat_input("Ask anything about your document…")

    user_question = typed_question or st.session_state.pending_question
    st.session_state.pending_question = None

    if user_question:

        user_message = {"role": "user", "content": user_question}
        st.session_state.messages.append(user_message)
        render_message(user_message)

        with st.chat_message("assistant"):

            with st.spinner("Searching your document…"):

                try:
                    answer, sources = answer_question(user_question)

                    assistant_message = {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    }

                except Exception as error:
                    assistant_message = {
                        "role": "assistant",
                        "content": (
                            "I couldn't answer that. Please try again.\n\n"
                            f"Details: {error}"
                        )
                    }

        st.session_state.messages.append(assistant_message)
        st.rerun()
