"""
preprocess.py
-------------
STEP 1 of the MedBot pipeline.

Reads every PDF and TXT file inside the `data/` folder, extracts the raw
text, cleans it, splits it into overlapping chunks, and saves the chunks
to `vector_store/chunks.json`.

Usage:
    python src/preprocess.py
"""

import os
import sys

from pypdf import PdfReader

# Allow `python src/preprocess.py` to find the sibling utils module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (DATA_DIR, CHUNKS_FILE, clean_text,
                   split_into_chunks, save_json)


# ---------------------------------------------------------------------------
# TEXT EXTRACTION
# ---------------------------------------------------------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from every page of a PDF file using pypdf.

    Pages that contain no extractable text (e.g. scanned images) simply
    contribute an empty string and are skipped.
    """
    reader = PdfReader(pdf_path)
    pages_text = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages_text.append(page_text)
        else:
            print(f"  [warning] Page {page_number} has no extractable text.")

    return "\n".join(pages_text)


def extract_text_from_txt(txt_path: str) -> str:
    """Read a plain-text file (UTF-8, ignoring undecodable bytes)."""
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_all_documents(data_dir: str) -> list:
    """
    Walk the data directory and return a list of dictionaries:
        [{"source": filename, "text": extracted_text}, ...]
    Supports .pdf and .txt files; everything else is ignored.
    """
    documents = []

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Data folder not found: {data_dir}\n"
            "Create it and place your medical book (PDF/TXT) inside.")

    for filename in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, filename)
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".pdf":
            print(f"[+] Extracting PDF : {filename}")
            text = extract_text_from_pdf(path)
        elif ext == ".txt":
            print(f"[+] Reading TXT    : {filename}")
            text = extract_text_from_txt(path)
        else:
            print(f"[-] Skipping unsupported file: {filename}")
            continue

        text = clean_text(text)
        if text:
            documents.append({"source": filename, "text": text})
        else:
            print(f"  [warning] No usable text found in {filename}.")

    return documents


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("MedBot - Step 1: Document Preprocessing")
    print("=" * 60)

    # 1. Load every document from data/
    documents = load_all_documents(DATA_DIR)
    if not documents:
        print("\nNo documents found. Place a PDF or TXT file in the "
              "'data/' folder and run this script again.")
        return

    # 2. Split each document into overlapping chunks
    all_chunks = []
    for doc in documents:
        chunks = split_into_chunks(doc["text"])
        print(f"[+] {doc['source']}: {len(chunks)} chunks created")
        for chunk in chunks:
            all_chunks.append({"source": doc["source"], "text": chunk})

    # 3. Persist the chunks for the embedding step
    save_json(all_chunks, CHUNKS_FILE)
    print(f"\n[OK] Saved {len(all_chunks)} chunks -> {CHUNKS_FILE}")
    print("Next step: python src/create_embeddings.py")


if __name__ == "__main__":
    main()
