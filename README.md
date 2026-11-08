# 📄 Document Q&A Chatbot

Upload a PDF and ask questions about it in plain English. The app finds the most relevant parts of your document, answers using only that content, and shows the pages the answer came from.

**[Live demo][(https://pdfq-achatbot-a7nwwxcw6dsdkypjfstapp9.streamlit.app/)** ]· Built with Streamlit, LangChain and Google Gemini

![App screenshot](<img width="1353" height="621" alt="pdf_chatbot" src="https://github.com/user-attachments/assets/ac7b76de-4d61-4b3e-bdf7-7ce2d1195a1b" />
)

## Features

- **Chat with any PDF.** Upload a document and ask questions in natural language.
- **Grounded answers.** Responses are based on the retrieved document content, and the bot says so when the answer isn't in the PDF.
- **Page sources.** Every answer includes a collapsible list of the PDF pages and text snippets it used.
- **Follow-up questions.** The last few messages are passed to the model, so you can ask "what about the second one?" and it understands.
- **Suggested questions.** One-click starters such as "Summarize this document".
- **Clean dark interface.** A custom-styled Streamlit UI with a document bar, chat bubbles and a floating input.
- **Session-only storage.** Your PDF is processed in memory and is not saved on the server.

## How it works

This app uses Retrieval-Augmented Generation (RAG):

1. **Load.** The PDF is read page by page with `PyPDFLoader`.
2. **Split.** The text is cut into chunks of 1,000 characters with 200 characters of overlap, so ideas aren't broken at the edges.
3. **Embed and index.** Each chunk is converted to a vector with Google's `gemini-embedding-001` and stored in an in-memory vector store.
4. **Retrieve.** For each question, the 4 most similar chunks are found.
5. **Generate.** The chunks, the recent chat history and the question go to a Gemini model, which writes the answer.
6. **Cite.** The pages of the retrieved chunks are shown under the answer.

```
PDF → pages → chunks → embeddings → vector store
                                         │
Question → embedding → top-4 chunks ─────┤
                                         ▼
                     chunks + chat history + question → Gemini → answer + sources
```

## Tech stack

| Area | Tools |
|---|---|
| Interface | Streamlit with custom CSS |
| Orchestration | LangChain |
| LLM | Google Gemini (Flash-Lite) |
| Embeddings | Google `gemini-embedding-001` |
| Vector store | LangChain `InMemoryVectorStore` |
| PDF parsing | PyPDF |
| Language | Python |

## Run it locally

**1. Clone the repository**

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd YOUR-REPO
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Add your Gemini API key**

Get a free key from [Google AI Studio](https://aistudio.google.com/), then set it as an environment variable.

```bash
# macOS / Linux
export GOOGLE_API_KEY="your_key_here"

# Windows (PowerShell)
$env:GOOGLE_API_KEY="your_key_here"
```

**4. Start the app**

```bash
streamlit run app.py
```

Open the link shown in the terminal, usually `http://localhost:8501`.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and create a new app from the repo, with `app.py` as the main file.
3. Under **Advanced settings → Secrets**, add:
   ```
   GOOGLE_API_KEY = "your_key_here"
   ```
4. Click **Deploy**.

## Project structure

```
├── app.py                  # Streamlit app: UI, PDF processing and RAG logic
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Dark theme settings
└── README.md
```

## Limitations

- Works with text-based PDFs. Scanned documents without a text layer need OCR first.
- The vector store lives in memory, so uploading a new PDF or restarting the app clears it.
- Answer quality depends on how well the retrieved chunks match the question.

## Ideas for the future

- Support multiple PDFs and other formats such as DOCX and TXT
- OCR for scanned documents
- Streaming answers as they are generated
- A persistent vector database such as Chroma or FAISS
- Chat export as a PDF or text file

