"""
create_embeddings.py
--------------------
STEP 2 of the MedBot pipeline.

Loads the text chunks produced by `preprocess.py`, converts each chunk
into a dense vector using a Sentence-Transformers model, normalises the
vectors, and stores them in a FAISS index for fast similarity search.

Usage:
    python src/create_embeddings.py
"""

import os
import sys

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (CHUNKS_FILE, FAISS_INDEX_FILE, EMBEDDING_MODEL_NAME,
                   VECTOR_STORE_DIR, load_json)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index using inner product (dot product) similarity.

    Because every embedding is L2-normalised before being added, the inner
    product between two vectors equals their cosine similarity. This lets
    us interpret search scores directly as cosine similarities in [−1, 1].
    """
    dimension = embeddings.shape[1]           # e.g. 384 for MiniLM-L6-v2
    index = faiss.IndexFlatIP(dimension)      # exact (brute-force) search
    index.add(embeddings)                     # add all chunk vectors
    return index


def main():
    print("=" * 60)
    print("MedBot - Step 2: Embedding Creation")
    print("=" * 60)

    # 1. Load the preprocessed chunks
    if not os.path.exists(CHUNKS_FILE):
        print("chunks.json not found. Run 'python src/preprocess.py' first.")
        return

    chunks = load_json(CHUNKS_FILE)
    texts = [c["text"] for c in chunks]
    print(f"[+] Loaded {len(texts)} chunks from {CHUNKS_FILE}")

    # 2. Load the embedding model (downloads automatically on first run)
    print(f"[+] Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # 3. Encode all chunks into vectors.
    #    normalize_embeddings=True L2-normalises each vector so that the
    #    inner-product search in FAISS becomes cosine similarity.
    print("[+] Encoding chunks (this may take a minute on CPU)...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")  # FAISS requires float32

    # 4. Build and save the FAISS index
    index = build_faiss_index(embeddings)
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_FILE)

    print(f"\n[OK] FAISS index with {index.ntotal} vectors "
          f"(dim={embeddings.shape[1]}) saved -> {FAISS_INDEX_FILE}")
    print("Next step: python src/gui.py")


if __name__ == "__main__":
    main()
