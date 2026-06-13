# Project Report: MedBot — A Domain-Specific Medical Chatbot using Retrieval-Augmented Generation

**Course:** Artificial Intelligence — Course Project
**Project:** Project 01 — Domain-Specific AI Chatbot using Fine-Tuned Language Models
**Domain:** Medical Assistant (MedBot)

---

## 1. Introduction

Professionals and students in specialised fields such as medicine, law, and finance
must work with large volumes of dense, technical material. A medical graduate beginning
a house job, for example, has the foundational knowledge but is often overwhelmed by the
sheer volume of terminology, procedures, and guidelines. A chatbot that can answer
questions directly from authoritative medical material is therefore a genuinely useful
assistant: it provides targeted information on demand and improves knowledge retention.

This project implements **MedBot**, a domain-specific chatbot that answers questions
using the contents of a chosen medical book. Rather than the traditional approach of
fine-tuning a BERT model for question answering, MedBot uses **Retrieval-Augmented
Generation (RAG)**. This modern approach is more practical for a university setting:
it runs on a normal laptop without a GPU, requires no expensive training, and is easy
to update — adding new knowledge is simply a matter of dropping a new document into a
folder and re-running the indexing step.

---

## 2. Objectives

The objectives of this project are to:

1. Build a chatbot that answers questions **restricted to a specific domain** (medicine).
2. Construct a **custom knowledge base** from a domain-specific book.
3. Implement the complete pipeline in **Python** with **well-commented, modular code**.
4. Cover **dataset preprocessing, embedding creation, retrieval, generation, and a GUI**.
5. Detect out-of-domain questions and respond with a clear fallback message.
6. Provide a clean, user-friendly **web interface** with PDF upload.

---

## 3. Dataset Description

The dataset is a **self-created medical knowledge document** placed in the `data/` folder.
The project accepts both `.pdf` and `.txt` files, so any medical textbook or set of notes
can be used. For demonstration and testing, a sample file `medical_book.txt` is included.
It contains introductory human-physiology material organised into sections:

- **Homeostasis** — the body's regulation of its internal environment.
- **Cardiovascular system** — the heart, blood vessels, blood pressure, and hypertension.
- **Respiratory system** — gas exchange in the alveoli.
- **Digestive system** — digestion, absorption, and the roles of the liver and pancreas.
- **Insulin and blood glucose** — the function of insulin and the basis of diabetes.
- **Nervous system** — the central and peripheral nervous systems and neurons.
- **Common symptoms and first aid** — fever, dehydration, and minor burns.

During preprocessing the document is cleaned and split into **overlapping text chunks**
(approximately 180 words each, with a 40-word overlap). The overlap ensures that sentences
falling on a chunk boundary still appear intact in at least one chunk, which improves
retrieval accuracy. To use a real textbook, the user simply replaces the sample file with
their own PDF/TXT and re-runs the pipeline.

---

## 4. Methodology

MedBot follows the **Retrieval-Augmented Generation** paradigm, which separates *knowledge*
(stored in a searchable index) from *language ability* (provided by a generative model).
The end-to-end methodology is:

1. **Document ingestion** — load PDF/TXT files and extract their raw text.
2. **Cleaning** — normalise whitespace and remove control characters.
3. **Chunking** — split text into overlapping word-based chunks.
4. **Embedding** — convert each chunk into a dense vector with a Sentence-Transformers model.
5. **Indexing** — store all vectors in a FAISS index for fast similarity search.
6. **Retrieval** — for a user question, embed it and fetch the top-K most similar chunks.
7. **Relevance gating** — if the best similarity score is below a threshold, declare the
   question out-of-domain and return the standard fallback message.
8. **Generation** — feed the retrieved chunks plus the question to FLAN-T5, which writes
   an answer grounded in that context.

Because embeddings are L2-normalised, the FAISS inner-product search is mathematically
equivalent to **cosine similarity**, allowing retrieval scores to be interpreted directly
as similarity values in the range [−1, 1].

---

## 5. Model Architecture

The system combines three free, open-source components:

| Component        | Model / Library                              | Role                                            |
|------------------|----------------------------------------------|-------------------------------------------------|
| Embedding model  | `sentence-transformers/all-MiniLM-L6-v2`     | Maps text to 384-dimensional semantic vectors.  |
| Vector store     | FAISS `IndexFlatIP`                          | Exact cosine-similarity search over all chunks. |
| Generative model | `google/flan-t5-large` (Hugging Face)        | Instruction-tuned model that writes the answer. |
| Interface        | Flask web app (HTML/CSS/JS)                  | Browser-based chat with PDF upload.             |

**Why these choices?**

- **MiniLM-L6-v2** is small (~80 MB), fast on CPU, and a strong general-purpose sentence
  encoder — ideal for a laptop project.
- **FAISS** is the industry standard for efficient vector similarity search.
- **FLAN-T5** is instruction-tuned, so it follows the "answer using only this context"
  prompt reliably. The `large` size gives noticeably fuller answers and runs on CPU within
  a 16GB machine; it can be swapped for `flan-t5-base` or `flan-t5-small` on lighter
  hardware without any code changes.

---

## 6. Implementation Details

The code is organised into well-commented modules inside `src/`:

- **`utils.py`** — central configuration (paths, model names, hyper-parameters) and helper
  functions for text cleaning, chunking, question normalisation, and JSON I/O. Keeping all
  settings in one place makes the project easy to tune.
- **`preprocess.py`** — text extraction (using `pypdf` for PDFs) and chunking helpers,
  reused both by the optional offline pipeline and by the web app's live ingestion.
- **`create_embeddings.py`** — an optional offline script that builds a FAISS index from the
  `data/` folder ahead of time.
- **`chatbot.py`** — the `MedBot` class. It loads the embedding and generation models once,
  then can build its knowledge base either from a saved index or, more importantly, **from an
  uploaded PDF at runtime** via `index_pdf()`. For each question it performs retrieval, the
  relevance check, prompt construction, and generation.
- **`app.py`** — the Flask web server. It creates a single `MedBot` at startup and exposes
  three routes: `/upload` (receive a PDF and rebuild the knowledge base), `/chat` (answer a
  question), and `/status` (report whether a document is loaded).
- **`templates/index.html`** — the browser chat interface (HTML, CSS, and vanilla
  JavaScript).

**Query normalisation.** Before a question is embedded for retrieval, conversational filler
and greetings (e.g. "hey can you explain me …") are stripped. These tokens otherwise pull the
question's embedding away from the clean textbook text, lowering similarity scores and
sometimes wrongly triggering the out-of-domain guard. The original question is still shown to
the generative model so answers read naturally.

**Key design decisions:**

- *Separation of offline and online work.* Heavy indexing is done once (steps 1–2);
  answering questions at runtime is fast.
- *Out-of-domain guard.* A configurable `SIMILARITY_THRESHOLD` keeps MedBot honest — it
  refuses to answer questions the book does not cover, exactly as required.
- *Grounded prompting.* The model is explicitly instructed to use only the supplied context,
  which reduces hallucination and keeps answers domain-specific.

---

## 7. Web Interface Description

The interface is a **single-page web application** served by Flask and rendered in the
browser, giving the familiar feel of a modern LLM chat assistant. It provides:

- **PDF upload** — a drag-and-drop area (or click-to-browse) that accepts PDF files only.
  Once a file is uploaded it is processed into the knowledge base on the fly, and a
  confirmation message reports how many sections were indexed. A paperclip button lets the
  user swap in a different PDF at any time, instantly rebuilding the knowledge base.
- **Chat window** — a scrollable conversation history with distinct styling for the user
  and the assistant. Out-of-domain ("not found") replies are highlighted so they stand out.
- **Input box and Send button** — pressing Enter sends a message (Shift+Enter inserts a new
  line); an animated "typing" indicator shows while the model generates an answer.
- **Document status indicator** — a chip in the header shows which document is currently
  loaded, with a small live indicator once the knowledge base is ready.

On the server side, the heavy models are loaded only once when the app starts. Uploading a
new PDF only rebuilds the lightweight FAISS index, so switching documents is fast and does
not reload the models. Uploaded files are handled from a temporary path and not stored
permanently.

---

## 8. Results

The pipeline was verified end-to-end on the sample knowledge base:

- **Preprocessing** correctly split the document into overlapping chunks and saved them.
- **Indexing** produced a FAISS index whose vector count matched the number of chunks.
- **Retrieval** returned the relevant passages for in-domain questions. For example,
  *"What is insulin and how does it affect blood glucose?"* retrieved the insulin/diabetes
  section, and *"the four chambers of the heart"* retrieved the cardiovascular section.
- **Out-of-domain detection** worked as intended: a question such as *"How do I change a
  car tyre?"* scored below the similarity threshold and produced the fallback message
  *"I could not find relevant information in the knowledge base."*
- **Generation** with FLAN-T5 produced concise answers grounded in the retrieved context.

These results confirm that MedBot answers domain questions accurately while correctly
refusing questions outside its knowledge base.

---

## 9. Conclusion

This project successfully delivers a domain-specific medical chatbot built on a modern
Retrieval-Augmented Generation architecture. By combining Sentence-Transformers embeddings,
a FAISS vector store, and an instruction-tuned FLAN-T5 generator, MedBot provides accurate,
context-grounded answers from a custom medical book while running entirely on a normal
laptop with free models. A clean web-based chat interface with PDF upload makes it easy to
use, and the modular, well
commented code base satisfies the academic requirements of the assignment. The RAG approach
proved both more practical and more maintainable than traditional BERT fine-tuning.

---

## 10. Future Improvements

- **Conversational memory** — track previous turns so follow-up questions ("and what about
  Type 2?") are understood in context.
- **Source citations** — display which chunk/section an answer came from, increasing trust.
- **Larger / multiple books** — index several medical references and scale FAISS with an
  approximate index (e.g. IVF/HNSW) for speed.
- **Better generation** — use a larger or more recent instruction model, or a local quantised
  LLM, for richer answers.
- **OCR support** — add OCR so scanned (image-only) medical PDFs can also be ingested.
- **Multi-user deployment** — host the Flask app on a server with per-user sessions so each
  user can work with their own uploaded document simultaneously.
