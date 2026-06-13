"""
chatbot.py
----------
The Retrieval-Augmented Generation (RAG) engine for MedBot.

Pipeline for every user question:
    1. Encode the question with the same Sentence-Transformers model
       used for the document chunks.
    2. Search the FAISS index for the TOP_K most similar chunks.
    3. If the best similarity score is below SIMILARITY_THRESHOLD,
       return the standard "not found" message (out-of-domain guard).
    4. Otherwise, build a prompt with the retrieved context and the
       question, and let FLAN-T5 generate a grounded answer.

The MedBot class loads its (heavy) models once. Its knowledge base can be
built two ways:
    * from a pre-built FAISS index on disk (the offline pipeline), or
    * on the fly from an uploaded PDF via index_pdf() - used by the web app.

This file can also be run directly for a command-line chat session:
    python src/chatbot.py
"""

import os
import sys

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (CHUNKS_FILE, FAISS_INDEX_FILE, EMBEDDING_MODEL_NAME,
                   GENERATION_MODEL_NAME, TOP_K, SIMILARITY_THRESHOLD,
                   MAX_NEW_TOKENS, MIN_NEW_TOKENS, NUM_BEAMS,
                   NOT_FOUND_MESSAGE, DEBUG_RETRIEVAL,
                   load_json, normalise_question, split_into_chunks,
                   clean_text)

# extract_text_from_pdf lives in preprocess.py; importing it here avoids
# duplicating the PDF-reading logic. (preprocess does not import chatbot,
# so there is no circular import.)
from preprocess import extract_text_from_pdf


class MedBot:
    """
    Domain-specific RAG chatbot.

    The embedding and generation models are loaded once in the constructor.
    The knowledge base (FAISS index + chunks) can either be loaded from disk
    or built at runtime from an uploaded PDF.
    """

    def __init__(self, load_existing: bool = True):
        # --- Load the heavy models once (this is the slow part) ----------
        print(f"[MedBot] Loading embedding model: {EMBEDDING_MODEL_NAME}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

        print(f"[MedBot] Loading generation model: {GENERATION_MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL_NAME)

        # --- Knowledge base (filled in below or via index_pdf) -----------
        self.index = None      # FAISS index
        self.chunks = []       # list of {"source", "text"} dicts
        self.source_name = None  # name of the loaded document

        # Optionally pick up a knowledge base built by the offline pipeline
        if load_existing and os.path.exists(CHUNKS_FILE) \
                and os.path.exists(FAISS_INDEX_FILE):
            print("[MedBot] Loading existing knowledge base from disk...")
            self.chunks = load_json(CHUNKS_FILE)
            self.index = faiss.read_index(FAISS_INDEX_FILE)
            self.source_name = "existing knowledge base"

        print("[MedBot] Ready.\n")

    # ------------------------------------------------------------------
    # KNOWLEDGE BASE
    # ------------------------------------------------------------------
    def has_knowledge(self) -> bool:
        """True once a document has been indexed and is ready to query."""
        return self.index is not None and self.index.ntotal > 0

    def _build_index(self, chunk_dicts: list):
        """Embed chunks and build an in-memory FAISS index from them."""
        texts = [c["text"] for c in chunk_dicts]
        embeddings = self.embedder.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,  # makes inner product == cosine sim
        ).astype("float32")

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        self.index = index
        self.chunks = chunk_dicts

    def index_pdf(self, pdf_path: str) -> int:
        """
        Build the knowledge base from a single uploaded PDF.

        Extracts text, cleans and chunks it, embeds the chunks, and builds a
        fresh FAISS index that replaces any previous one.

        Returns
        -------
        int : the number of chunks indexed.

        Raises
        ------
        ValueError : if no readable text could be extracted from the PDF.
        """
        raw_text = extract_text_from_pdf(pdf_path)
        text = clean_text(raw_text)
        if not text:
            raise ValueError(
                "No readable text found in this PDF. It may be a scanned "
                "(image-only) document, which would need OCR.")

        source = os.path.basename(pdf_path)
        chunk_dicts = [{"source": source, "text": c}
                       for c in split_into_chunks(text)]
        if not chunk_dicts:
            raise ValueError("The PDF text was too short to build a "
                             "knowledge base.")

        self._build_index(chunk_dicts)
        self.source_name = source

        if DEBUG_RETRIEVAL:
            print(f"[index_pdf] Extracted {len(text):,} chars from '{source}', "
                  f"built {len(chunk_dicts)} chunks.")

        return len(chunk_dicts)

    # ------------------------------------------------------------------
    # RETRIEVAL
    # ------------------------------------------------------------------
    def retrieve(self, question: str):
        """
        Return (top_chunks, best_score) for a question.

        top_chunks : list[str]  - the TOP_K most similar chunk texts
        best_score : float      - cosine similarity of the best match
        """
        # Strip greetings/filler so the query embeds like clean book text.
        clean_query = normalise_question(question)
        query_vector = self.embedder.encode(
            [clean_query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        # Never ask FAISS for more neighbours than it actually holds.
        k = min(TOP_K, self.index.ntotal)
        scores, ids = self.index.search(query_vector, k)
        scores, ids = scores[0], ids[0]

        top_chunks = [self.chunks[i]["text"] for i in ids if i != -1]
        best_score = float(scores[0]) if len(scores) else 0.0
        return top_chunks, best_score

    # ------------------------------------------------------------------
    # GENERATION
    # ------------------------------------------------------------------
    def build_prompt(self, question: str, context_chunks: list) -> str:
        """Combine retrieved chunks and the question into a FLAN-T5 prompt.

        The "not found" guard lives entirely in answer() via the similarity
        threshold — the model is never told to emit that phrase, so it cannot
        echo it back as a false negative.
        """
        context = "\n\n".join(context_chunks)
        prompt = (
            "You are MedBot, a knowledgeable medical assistant. "
            "Using ONLY the context below, answer the question in 2 to 4 "
            "complete prose sentences. "
            # Explicitly forbid list/colon lead-ins — FLAN-T5 defaults to
            # them and then halts after the colon, producing truncated output.
            "Do NOT use bullet points, numbered lists, or colons. "
            "Do NOT begin with phrases like 'The following are' or "
            "'There are several'. State the information directly.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        return prompt

    def answer(self, question: str) -> str:
        """Full RAG pipeline for a single question. Always returns a string."""
        question = question.strip()
        if not question:
            return "Please type a question."

        # Guard: no document loaded yet
        if not self.has_knowledge():
            return ("Please upload a PDF first so I have a knowledge base "
                    "to answer from.")

        # 1) Retrieve relevant knowledge
        context_chunks, best_score = self.retrieve(question)

        # Diagnostics — printed to the server console whenever DEBUG_RETRIEVAL
        # is True.  Shows the score that drives the threshold decision so you
        # can see exactly why a question passes or fails the guard.
        if DEBUG_RETRIEVAL:
            preview = context_chunks[0][:120].replace("\n", " ") if context_chunks else "—"
            print(
                f"[retrieval] score={best_score:.4f}  threshold={SIMILARITY_THRESHOLD}"
                f"  chunks={len(context_chunks)}\n"
                f"[retrieval] top-chunk preview: {preview!r}"
            )

        # 2) Out-of-domain guard — sole source of NOT_FOUND_MESSAGE.
        # The generation prompt never mentions this phrase, so the model
        # cannot echo it as a spurious "not found" on in-domain questions.
        if best_score < SIMILARITY_THRESHOLD or not context_chunks:
            return NOT_FOUND_MESSAGE

        # 3) Generate an answer grounded in the retrieved context
        prompt = self.build_prompt(question, context_chunks)
        inputs = self.tokenizer(prompt, return_tensors="pt",
                                max_length=512, truncation=True)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            # Force at least MIN_NEW_TOKENS tokens so the model cannot stop
            # immediately after a colon lead-in ("The symptoms are:").
            min_new_tokens=MIN_NEW_TOKENS,
            # Beam search produces more complete sentences than greedy decoding.
            num_beams=NUM_BEAMS,
            # Prevent the model from repeating the same 3-word phrase, which
            # also discourages it from looping on list-style patterns.
            no_repeat_ngram_size=3,
            # Stop as soon as all beams have hit an EOS token (avoids padding).
            early_stopping=True,
        )
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # 4) Safety net for empty / unhelpful generations
        if not answer or answer.lower() in {"i don't know", "i do not know",
                                            "unknown", "no answer"}:
            return NOT_FOUND_MESSAGE
        return answer


# ---------------------------------------------------------------------------
# Optional command-line interface (handy for quick testing)
# ---------------------------------------------------------------------------
def main():
    bot = MedBot()
    if not bot.has_knowledge():
        print("No knowledge base on disk. Run the web app (python src/app.py) "
              "and upload a PDF, or run the offline pipeline first.")
        return
    print("Type a medical question (or 'quit' to exit).")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        print("MedBot:", bot.answer(question))


if __name__ == "__main__":
    main()
