# Import standard libraries
import os                # For interacting with the operating system (not heavily used here)
import json              # For reading/writing JSON files
import hashlib           # For generating file hashes (MD5)
from pathlib import Path # For handling file paths in a clean, cross-platform way
from typing import List, Dict  # For type hints

# Import third-party libraries
import pdfplumber        # For extracting text from PDF files
import docx              # For extracting text from Word (.docx) files
from sentence_transformers import SentenceTransformer  # For embedding text into vectors
import faiss             # Facebook AI Similarity Search library (vector index for semantic search)

# -----------------------------
# Define directories for pipeline
# -----------------------------
SOURCE_DIR = Path("data/source")       # Raw input files (PDF, DOCX, TXT, etc.)
PROCESSED_DIR = Path("data/processed") # Cleaned text versions of source files
CHUNKS_DIR = Path("data/chunks")       # Text chunks split into smaller pieces
EMBEDDINGS_DIR = Path("data/embeddings") # Embeddings (vector representations of chunks)
INDEX_DIR = Path("data/index")         # FAISS index + metadata

# Ensure required directories exist (create them if missing)
for d in [PROCESSED_DIR, CHUNKS_DIR, EMBEDDINGS_DIR, INDEX_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load embedding model
# -----------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight, fast embedding model
embedder = SentenceTransformer(MODEL_NAME)

# -----------------------------
# Utility functions
# -----------------------------

def hash_file(path: Path) -> str:
    """Generate MD5 hash of a file (used for deduplication or tracking)."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def extract_text(path: Path) -> str:
    """Extract text from PDF, DOCX, TXT, or MD files."""
    if path.suffix.lower() == ".pdf":
        text = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                # Extract text page by page, prefix with page number
                text.append(f"[Page {i}]\n{page.extract_text() or ''}")
        return "\n".join(text)

    elif path.suffix.lower() == ".docx":
        doc = docx.Document(path)
        # Extract text from each paragraph
        return "\n".join([p.text for p in doc.paragraphs])

    elif path.suffix.lower() in [".txt", ".md"]:
        # Read plain text or markdown files
        return path.read_text(encoding="utf-8", errors="ignore")

    else:
        # Unsupported file type
        raise ValueError(f"Unsupported file type: {path}")


def save_processed(path: Path, text: str):
    """Save extracted text into processed directory as .txt file."""
    out = PROCESSED_DIR / (path.stem + ".txt")
    out.write_text(text, encoding="utf-8")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks.
    - chunk_size: number of words per chunk
    - overlap: number of words overlapping between consecutive chunks
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        # Move start forward but keep overlap
        start += chunk_size - overlap
    return chunks


def build_chunks(path: Path, text: str) -> List[Dict]:
    """Build chunk metadata for a file."""
    chunks = chunk_text(text)
    return [
        {
            "chunk_id": f"{path.stem}_{i}",  # Unique ID per chunk
            "source_path": str(path),        # Original file path
            "text": chunk                    # Chunk text
        }
        for i, chunk in enumerate(chunks)
    ]


def save_chunks(path: Path, chunks: List[Dict]):
    """Save chunks into JSONL file (one JSON object per line)."""
    out = CHUNKS_DIR / (path.stem + ".jsonl")
    with out.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")


def embed_chunks(chunks: List[Dict]) -> List[List[float]]:
    """Generate embeddings for each chunk using the sentence transformer model."""
    texts = [c["text"] for c in chunks]
    return embedder.encode(texts, convert_to_numpy=True)


def build_index(all_chunks: List[Dict], all_embeddings):
    """Build FAISS index from embeddings and save metadata."""
    dim = all_embeddings.shape[1]  # Embedding dimension
    index = faiss.IndexFlatL2(dim) # L2 distance index
    index.add(all_embeddings)      # Add embeddings to index
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))

    # Save metadata (chunk info)
    meta = {c["chunk_id"]: c for c in all_chunks}
    (INDEX_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


# -----------------------------
# Main ingestion pipeline
# -----------------------------
def ingest():
    """Process all source files: extract text, chunk, embed, and build index."""
    all_chunks = []
    all_embeddings = []

    # Iterate through all files in source directory
    for file in SOURCE_DIR.iterdir():
        if file.is_file():
            print(f"Processing {file.name}...")
            text = extract_text(file)        # Extract text
            save_processed(file, text)       # Save processed text

            chunks = build_chunks(file, text) # Build chunks
            save_chunks(file, chunks)         # Save chunks
            all_chunks.extend(chunks)         # Add to global list

            embeddings = embed_chunks(chunks) # Generate embeddings
            all_embeddings.extend(embeddings) # Add to global list

    # Convert embeddings list to numpy array
    import numpy as np
    all_embeddings = np.array(all_embeddings)

    # Build FAISS index and save metadata
    build_index(all_chunks, all_embeddings)
    print("✅ Ingestion complete: processed, chunks, embeddings, index built.")


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    ingest()
