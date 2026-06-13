# MedBot — Medical RAG Chatbot

A domain-specific medical chatbot that answers questions from any PDF you upload.
Built with **Retrieval-Augmented Generation (RAG)**: it retrieves the most relevant
passages from the document and feeds them to a generative model, so every answer is
grounded in the uploaded content rather than hallucinated.

Runs entirely on CPU — no GPU, no paid API keys, no cloud services required.

---

## How It Works

```
            ┌─────────────── browser (chat UI) ───────────────┐
            │  upload PDF                 ask a question       │
            └──────┬──────────────────────────────┬───────────┘
                   ▼                               ▼
          extract + clean + chunk            embed question
                   ▼                               ▼
            embed chunks ──▶ FAISS index ──▶ retrieve top-K chunks
                                                   ▼
                                       FLAN-T5 generates answer
```

| Stage      | Model / Library                                      |
|------------|------------------------------------------------------|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`             |
| Retrieval  | FAISS — cosine similarity, exact nearest-neighbour   |
| Generation | `google/flan-t5-large` (Hugging Face Transformers)   |
| Web server | Flask 3                                              |

All models download automatically on first run and are cached locally.

---

## Project Structure

```
MedBot/
├── data/
│   └── medical_book.txt        # sample reference document
├── models/                     # placeholder for locally saved models
├── vector_store/               # FAISS index + chunks (built at runtime)
├── templates/
│   └── index.html              # single-page chat UI
├── src/
│   ├── utils.py                # all config constants + helper functions
│   ├── preprocess.py           # PDF text extraction + cleaning
│   ├── create_embeddings.py    # optional offline index builder
│   ├── chatbot.py              # RAG engine: retrieval + generation
│   └── app.py                  # Flask web server  ← entry point
├── requirements.txt
├── report.md
└── README.md
```

---

## Requirements

- Python 3.9 – 3.12
- ~3 GB disk space for the FLAN-T5-large model (downloaded once)
- ~4 GB RAM recommended

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Hassan-Younas18/RAG-System.git
cd RAG-System
```

### 2. Create a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows (Command Prompt)
venv\Scripts\activate.bat

# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

> **Windows users:** PyTorch contains deeply nested file paths that exceed
> Windows' default 260-character limit. If `pip install` fails with an
> `OSError: [Errno 2] No such file or directory` hint about long paths, either:
> - Enable Long Path support in Group Policy (`gpedit.msc` → Computer Configuration →
>   Administrative Templates → System → Filesystem → Enable Win32 long paths), **or**
> - Create the virtual environment in a short root path to keep total path lengths
>   under the limit:
>   ```
>   python -m venv C:\venv
>   C:\venv\Scripts\activate
>   ```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the App

Start the web server from the project root:

```bash
python src/app.py
```

Then open **http://127.0.0.1:5000** in your browser.

> **First launch** downloads the embedding model (~90 MB) and the generation model
> (~3 GB for FLAN-T5-large). This takes a few minutes and requires an internet
> connection once. All subsequent launches load from the local cache and start in
> under 30 seconds.

---

## Usage

1. **Upload a PDF** — drag it onto the upload card or click *Choose file*.
   Only text-based PDFs are supported (scanned/image PDFs have no extractable text).
2. Wait for the confirmation: *"'filename.pdf' is ready. Indexed N sections."*
3. **Ask a question** in the chat box and press **Enter** (Shift+Enter for a new line).
4. **Switch documents** any time — click the paperclip icon to upload a new PDF and
   rebuild the knowledge base instantly.

If a question falls outside the content of the uploaded document, MedBot replies:

> "I could not find relevant information in the knowledge base."

---

## Configuration (`src/utils.py`)

All tunable parameters live in one place:

| Constant             | Default                    | Effect                                              |
|----------------------|----------------------------|-----------------------------------------------------|
| `GENERATION_MODEL_NAME` | `google/flan-t5-large`  | Swap for `flan-t5-base` / `flan-t5-small` to reduce RAM and speed up inference |
| `SIMILARITY_THRESHOLD`  | `0.15`                  | Raise to make the out-of-domain guard stricter; lower if too many questions are rejected |
| `TOP_K`                 | `6`                     | Number of chunks retrieved per question             |
| `CHUNK_SIZE`            | `180`                   | Approximate words per chunk (changing this requires re-indexing) |
| `CHUNK_OVERLAP`         | `40`                    | Overlap between consecutive chunks                  |
| `MAX_NEW_TOKENS`        | `256`                   | Maximum answer length                               |
| `MIN_NEW_TOKENS`        | `40`                    | Minimum answer length — prevents premature cutoff   |
| `NUM_BEAMS`             | `4`                     | Beam search width — higher gives more complete answers at the cost of speed |
| `DEBUG_RETRIEVAL`       | `True`                  | Prints similarity scores and chunk previews to the server console for every question |

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `pip install` fails with `OSError` about a missing file | Windows long-path limit | See the Windows note in Installation step 2 |
| Answer is always "I could not find…" | `SIMILARITY_THRESHOLD` too high, or PDF text didn't extract | Lower threshold to `0.10`; check `[retrieval] score=` in the console |
| Answer is a truncated colon lead-in | `MIN_NEW_TOKENS` too low | Raise `MIN_NEW_TOKENS` in `utils.py` |
| Server crashes on startup | Port 5000 already in use | Change `port=5000` to another port (e.g. `5001`) in `src/app.py` |
| Uploaded PDF is rejected | Scanned/image-only PDF | The PDF must contain selectable text; OCR it first with a tool like Adobe Acrobat |
| Generation is very slow | FLAN-T5-large on CPU | Switch to `flan-t5-base` in `utils.py` for faster (but slightly weaker) responses |

---

## Optional: Offline Index Builder

If you want to pre-build the knowledge base from `data/medical_book.txt` instead of
uploading a PDF at runtime:

```bash
python src/create_embeddings.py
```

Then start the server with `load_existing=True` (already the default for the CLI mode
in `chatbot.py`):

```bash
python src/chatbot.py   # command-line chat using the pre-built index
```

---

## Acknowledgements

- [Sentence-Transformers](https://www.sbert.net/) — embedding model
- [FAISS](https://github.com/facebookresearch/faiss) — vector search
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) — FLAN-T5
- [Flask](https://flask.palletsprojects.com/) — web framework

---

*Course project — Artificial Intelligence, Batch 21*
