import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict

import pdfplumber
import docx
from sentence_transformers import SentenceTransformer
import faiss

# Paths
SOURCE_DIR = Path("data/source")
PROCESSED_DIR = Path("data/processed")
CHUNKS_DIR = Path("data/chunks")
EMBEDDINGS_DIR = Path("data/embeddings")
INDEX_DIR = Path("data/index")

# Ensure dirs exist
for d in [PROCESSED_DIR, CHUNKS_DIR, EMBEDDINGS_DIR, INDEX_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Embedding model (local, fast)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(MODEL_NAME)


def hash_file(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def extract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        text = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text.append(f"[Page {i}]\n{page.extract_text() or ''}")
        return "\n".join(text)
    elif path.suffix.lower() == ".docx":
        doc = docx.Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    elif path.suffix.lower() in [".txt", ".md"]:
        return path.read_text(encoding="utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type: {path}")


def save_processed(path: Path, text: str):
    out = PROCESSED_DIR / (path.stem + ".txt")
    out.write_text(text, encoding="utf-8")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def build_chunks(path: Path, text: str) -> List[Dict]:
    chunks = chunk_text(text)
    return [
        {
            "chunk_id": f"{path.stem}_{i}",
            "source_path": str(path),
            "text": chunk
        }
        for i, chunk in enumerate(chunks)
    ]


def save_chunks(path: Path, chunks: List[Dict]):
    out = CHUNKS_DIR / (path.stem + ".jsonl")
    with out.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")


def embed_chunks(chunks: List[Dict]) -> List[List[float]]:
    texts = [c["text"] for c in chunks]
    return embedder.encode(texts, convert_to_numpy=True)


def build_index(all_chunks: List[Dict], all_embeddings):
    dim = all_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(all_embeddings)
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))

    # Save metadata
    meta = {c["chunk_id"]: c for c in all_chunks}
    (INDEX_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def ingest():
    all_chunks = []
    all_embeddings = []

    for file in SOURCE_DIR.iterdir():
        if file.is_file():
            print(f"Processing {file.name}...")
            text = extract_text(file)
            save_processed(file, text)

            chunks = build_chunks(file, text)
            save_chunks(file, chunks)
            all_chunks.extend(chunks)

            embeddings = embed_chunks(chunks)
            all_embeddings.extend(embeddings)

    import numpy as np
    all_embeddings = np.array(all_embeddings)
    build_index(all_chunks, all_embeddings)
    print("✅ Ingestion complete: processed, chunks, embeddings, index built.")


if __name__ == "__main__":
    ingest()
