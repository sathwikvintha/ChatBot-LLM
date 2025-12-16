# Import standard libraries
import json                # For reading/writing JSON files
import numpy as np         # For numerical operations (arrays, embeddings)
from pathlib import Path   # For handling file paths in a clean way

# Import third-party libraries
from sentence_transformers import SentenceTransformer  # For embedding text into vectors
import faiss               # Facebook AI Similarity Search library (vector index)

# -----------------------------
# Define directories
# -----------------------------
CHUNK_DIR = Path("data/chunks")          # Directory containing chunked text files
INDEX_DIR = Path("data/faiss_index")     # Directory to store FAISS index + metadata
INDEX_DIR.mkdir(parents=True, exist_ok=True)  # Ensure index directory exists

# -----------------------------
# Load embedding model
# -----------------------------
# This model converts text into dense vector representations
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# -----------------------------
# Main function
# -----------------------------
def main():
    texts, metas = [], []   # Lists to store chunk texts and metadata

    # Iterate through all JSON files in chunk directory
    for jf in CHUNK_DIR.glob("*.json"):
        # Read JSON file and parse into Python object
        items = json.loads(jf.read_text(encoding="utf-8"))

        # Each item represents a chunk with metadata
        for it in items:
            texts.append(it["text"])   # Collect chunk text
            metas.append({             # Collect metadata for each chunk
                "chunk_id": it["chunk_id"],
                "doc_id": it["doc_id"],
                "source_path": it["source_path"],
                "page_number": it.get("page_number"),       # Optional
                "section_heading": it.get("section_heading"), # Optional
                "doc_type": it.get("doc_type")              # Optional
            })

    # -----------------------------
    # Generate embeddings
    # -----------------------------
    # Encode all chunk texts into vectors
    emb = model.encode(
        texts,
        batch_size=64,                # Process 64 texts at a time
        show_progress_bar=True,       # Show progress bar during encoding
        convert_to_numpy=True,        # Return numpy array
        normalize_embeddings=True     # Normalize vectors (important for similarity search)
    )

    # -----------------------------
    # Build FAISS index
    # -----------------------------
    dim = emb.shape[1]                # Embedding dimension
    index = faiss.IndexHNSWFlat(dim, 32)  # HNSW index (graph-based, efficient for large datasets)
    index.hnsw.efConstruction = 200       # Construction parameter (higher = more accurate, slower build)
    index.add(emb)                        # Add embeddings to index

    # Save FAISS index to disk
    faiss.write_index(index, str(INDEX_DIR / "index.faiss"))

    # Save metadata to disk (JSON file)
    (INDEX_DIR / "meta.json").write_text(
        json.dumps(metas, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # Print summary
    print(f"Indexed {len(texts)} chunks.")

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    main()
