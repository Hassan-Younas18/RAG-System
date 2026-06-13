# MedBot — Domain-Specific Medical Chatbot (RAG + Web UI)

MedBot is a domain-specific chatbot for the **Artificial Intelligence course project**.
Instead of classic BERT QA fine-tuning, it uses **Retrieval-Augmented Generation (RAG)**,
which runs comfortably on a normal laptop (CPU only) and is easy to reproduce.

You open it in your **web browser**, **upload a medical PDF**, and chat with it like a
normal LLM assistant. Your questions are answered using only the content of the uploaded
document. If nothing relevant is found, the bot replies:

> "I could not find relevant information in the knowledge base."

---

## How It Works (Pipeline)

```
            ┌─────────────── browser (chat UI) ───────────────┐
            │  upload PDF                 ask a question       │
            └──────┬──────────────────────────────┬───────────┘
                   ▼                               ▼
          extract + clean + chunk            embed question
                   ▼                               ▼
            embed chunks ─▶ FAISS index ─▶ retrieve top-K chunks
                                               ▼
                                   FLAN-T5 generates answer
```

| Stage         | Tool                                            |
|---------------|-------------------------------------------------|
| Embeddings    | Sentence-Transformers `all-MiniLM-L6-v2`        |
| Retrieval     | FAISS (cosine similarity, exact search)         |
| Generation    | Hugging Face Transformers `google/flan-t5-large`|
| Interface     | Flask web app (HTML/CSS/JS chat page)           |

All models are **free** and download automatically on first run.

---

## Folder Structure

```
MedBot/
├── data/                       # (optional) sample documents
│   └── medical_book.txt
├── models/                     # reserved for cached/local models
├── vector_store/               # used by the optional offline pipeline
├── templates/
│   └── index.html              # the chat web page
├── src/
│   ├── utils.py                # config + helper functions
│   ├── preprocess.py           # PDF/TXT loading + chunking helpers
│   ├── create_embeddings.py    # (optional) offline index builder
│   ├── chatbot.py              # RAG engine (retrieval + generation)
│   └── app.py                  # Flask web server  ← main entry point
├── requirements.txt
├── README.md
└── report.md
```

---

## Installation Guide

**1. Use Python 3.9–3.12 and (recommended) create a virtual environment:**

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

**2. Install the dependencies:**

```bash
pip install -r requirements.txt
```

---

## Execution Commands

Start the web server from the project root (`MedBot/`):

```bash
python src/app.py
```

Then open this address in your browser:

```
http://127.0.0.1:5000
```

Upload a medical PDF using the upload box (or drag & drop), wait a moment while it is
indexed, then start asking questions.

> The **first** launch downloads the embedding and generation models (a few hundred MB to
> ~3 GB for FLAN-T5-large). This needs an internet connection once; afterwards everything
> runs offline.

(Optional) A command-line tester is also available once a knowledge base exists:

```bash
python src/chatbot.py
```

---

## Using the Web App

1. **Upload** a PDF — drag it onto the upload card or click to browse (PDF only).
2. Wait for the confirmation message ("… is ready. Indexed N sections.").
3. **Ask questions** in the input box. Press **Enter** to send (Shift+Enter for a new line).
4. **Upload a different PDF** any time using the paperclip button — the knowledge base
   is rebuilt instantly from the new document.

Out-of-domain questions (not covered by the PDF) return the standard
"I could not find relevant information in the knowledge base." message.

---

## Tuning (src/utils.py)

- `CHUNK_SIZE`, `CHUNK_OVERLAP` — how documents are split.
- `TOP_K` — how many chunks are retrieved per question.
- `SIMILARITY_THRESHOLD` — how strict the out-of-domain guard is.
- `GENERATION_MODEL_NAME` — swap `flan-t5-large` for `flan-t5-base` (faster, lighter)
  or `flan-t5-small` on low-memory machines.

---

## Notes

- A scanned (image-only) PDF has no extractable text and will be rejected with a clear
  message; it would need OCR first.
- Uploaded PDFs are processed in a temporary file and not stored permanently.
