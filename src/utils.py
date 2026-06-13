"""
utils.py
--------
Shared configuration constants and helper functions used across the
MedBot project (preprocessing, embedding creation, chatbot, and GUI).

Author : Group XX
Course : Artificial Intelligence - Course Project
"""

import os
import re
import json

# ---------------------------------------------------------------------------
# 1. PROJECT PATHS
# ---------------------------------------------------------------------------
# Resolve all paths relative to the project root so the code works no matter
# which directory the scripts are launched from.
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
VECTOR_STORE_DIR = os.path.join(PROJECT_ROOT, "vector_store")

# Intermediate / output artefacts
CHUNKS_FILE = os.path.join(VECTOR_STORE_DIR, "chunks.json")
FAISS_INDEX_FILE = os.path.join(VECTOR_STORE_DIR, "faiss_index.bin")

# ---------------------------------------------------------------------------
# 2. MODEL CONFIGURATION
# ---------------------------------------------------------------------------
# Embedding model: small, fast, free, runs comfortably on CPU.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Generative model: FLAN-T5 is instruction-tuned and free.
# "large" gives noticeably fuller answers and needs ~3GB RAM (fine on 16GB).
# Drop to "base" (faster, weaker) or "small" if you are short on memory.
GENERATION_MODEL_NAME = "google/flan-t5-large"

# ---------------------------------------------------------------------------
# 3. RETRIEVAL / CHUNKING HYPER-PARAMETERS
# ---------------------------------------------------------------------------
CHUNK_SIZE = 180          # approximate number of words per chunk
CHUNK_OVERLAP = 40        # words shared between consecutive chunks
TOP_K = 6                 # number of chunks retrieved per query
SIMILARITY_THRESHOLD = 0.15  # cosine similarity below this = "not found"
MAX_NEW_TOKENS = 256      # maximum tokens the model may produce
# Minimum tokens to generate — prevents the model from stopping after a
# lead-in like "The following are the symptoms of angina:" with nothing after.
MIN_NEW_TOKENS = 40
# Beam search width: more beams → fuller, more coherent sentences at the
# cost of slightly slower inference. 4 is a good balance on CPU.
NUM_BEAMS = 4

# Standard fallback message required by the project specification.
NOT_FOUND_MESSAGE = "I could not find relevant information in the knowledge base."

# When True, answer() prints retrieval diagnostics (score, threshold,
# chunk count, top-chunk preview) for every question so you can tune
# SIMILARITY_THRESHOLD without guessing.
DEBUG_RETRIEVAL = True


# ---------------------------------------------------------------------------
# 4. HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Normalise raw text extracted from a PDF/TXT file.

    Steps:
      * collapse multiple whitespace characters into a single space
      * remove non-printable control characters
      * strip leading/trailing whitespace
    """
    # Remove control characters (keep normal punctuation and unicode letters)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    # Collapse all whitespace (newlines, tabs, multiple spaces) to one space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalise_question(question: str) -> str:
    """
    Clean a user question before it is embedded for retrieval.

    Conversational filler ("hey", "can you", "please explain me ...") adds
    tokens that pull the question's embedding away from the clean textbook
    text, which lowers similarity scores and can wrongly trigger the
    out-of-domain guard. We strip a small set of common lead-in phrases and
    greetings so "hey can you explain me the respiratory system?" embeds
    much like "the respiratory system".

    Note: this normalised text is used ONLY for retrieval. The original
    question is still shown to the generative model so the answer reads
    naturally.
    """
    q = question.strip().lower()

    # Remove leading greetings / politeness tokens
    q = re.sub(r"^(hey|hi|hello|yo|ok|okay|so|um|please)\b[\s,]*", " ", q)

    # Remove common lead-in request phrases anywhere near the start
    fillers = [
        r"can you ", r"could you ", r"would you ", r"will you ",
        r"please ", r"explain me ", r"explain ", r"tell me about ",
        r"tell me ", r"i want to know ", r"i'd like to know ",
        r"do you know ", r"what do you know about ",
    ]
    for f in fillers:
        q = re.sub(r"\b" + f, " ", q)

    q = re.sub(r"\s+", " ", q).strip()
    # If stripping removed everything, fall back to the original question
    return q if q else question.strip()


def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE,
                      overlap: int = CHUNK_OVERLAP) -> list:
    """
    Split a long text into overlapping word-based chunks.

    Overlap is used so that sentences cut at a chunk boundary still appear
    fully inside at least one chunk, which improves retrieval quality.

    Returns
    -------
    list[str] : list of text chunks
    """
    words = text.split()
    chunks = []
    step = chunk_size - overlap  # how far the window moves each iteration

    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        # Ignore tiny trailing fragments (less than 25 words)
        if len(chunk_words) < 25:
            continue
        chunks.append(" ".join(chunk_words))

    return chunks


def save_json(obj, path: str) -> None:
    """Save a Python object as a UTF-8 JSON file (creates folders if needed)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str):
    """Load and return a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
