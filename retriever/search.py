# Import standard libraries
import json   # For reading/writing JSON files
import re     # For regular expressions (used in tokenization)
from pathlib import Path   # For handling file paths in a clean way

# Import third-party libraries
import numpy as np   # For numerical operations (vectors, normalization, sorting)
from sentence_transformers import SentenceTransformer   # For embedding text into vectors
import faiss   # Facebook AI Similarity Search library (vector index for semantic search)
from rank_bm25 import BM25Okapi   # BM25 algorithm for keyword-based search

# Define paths for index and metadata
INDEX_DIR = Path("data/index")   # Directory where FAISS index and metadata are stored
META_PATH = INDEX_DIR / "meta.json"   # Path to metadata file (contains text chunks info)
FAISS_PATH = INDEX_DIR / "faiss.index"   # Path to FAISS index file

# Load the sentence transformer embedding model
# This model converts text into dense vector representations
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load the FAISS index from disk
# This index stores embeddings for fast similarity search
index = faiss.read_index(str(FAISS_PATH))

# Function to load metadata (chunk information) from meta.json
def _load_metas(path: Path):
    # Read JSON file as text and parse into Python object
    raw = json.loads(path.read_text(encoding="utf-8"))

    # Handle if metadata is stored as dict or list
    if isinstance(raw, dict):
        meta_list = list(raw.values())   # Convert dict values to list
    elif isinstance(raw, list):
        meta_list = raw   # Already a list
    else:
        raise ValueError("meta.json must be a list or a dict of chunk metadata")

    # Ensure each metadata entry has a unique chunk_id
    for i, m in enumerate(meta_list):
        if "chunk_id" not in m:
            # If no chunk_id, use 'id' field or fallback to "chunk_i"
            m["chunk_id"] = m.get("id", f"chunk_{i}")
    return meta_list

# Load metadata into memory
metas = _load_metas(META_PATH)

# Basic validation: check if FAISS index count matches metadata count
try:
    ntotal = index.ntotal   # Number of vectors stored in FAISS index
    if ntotal != len(metas):
        print(f"⚠️ Warning: FAISS index count ({ntotal}) != meta count ({len(metas)}). "
              "Retrieval may misalign if ingest ordering differs.")
except Exception:
    # If index fails to load, ignore silently
    pass

# Tokenization function for BM25
def _tokenize(text: str):
    # Convert text to lowercase and extract words (alphanumeric sequences)
    return re.findall(r"\b\w+\b", text.lower())

# Prepare BM25 corpus
corpus_texts = [m.get("text", "") for m in metas]   # Extract text from metadata
tokenized_corpus = [_tokenize(t) for t in corpus_texts]   # Tokenize each text
bm25 = BM25Okapi(tokenized_corpus)   # Initialize BM25 with tokenized corpus

# Function: semantic search using FAISS
def faiss_topk(query: str, k: int = 10):
    # Encode query into vector using sentence transformer
    q = model.encode([query], convert_to_numpy=True)

    # Normalize query vector (important for cosine similarity)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)

    # Perform search in FAISS index, get distances (D) and indices (I)
    D, I = index.search(q, k)

    results = []
    for score, idx in zip(D[0], I[0]):
        m = metas[idx]   # Get metadata for matched chunk
        # Invert distance so higher score = better similarity
        results.append({**m, "score": float(-score)})
    return results

# Function: keyword search using BM25
def bm25_topk(query: str, k: int = 10):
    # Tokenize query into words
    tokens = _tokenize(query)

    # Get BM25 scores for all documents
    scores = bm25.get_scores(tokens)

    # Get indices of top-k highest scores (sorted descending)
    idxs = np.argsort(scores)[-k:][::-1]

    results = []
    for i in idxs:
        m = metas[i]   # Get metadata for matched chunk
        results.append({**m, "bm25": float(scores[i])})
    return results

# Function: hybrid search (combine FAISS + BM25 results)
def hybrid_search(query: str, k: int = 10):
    # Get semantic results (FAISS)
    sem = faiss_topk(query, k)

    # Get keyword results (BM25)
    kw = bm25_topk(query, k)

    # Merge results by chunk_id
    merged = {r["chunk_id"]: r for r in sem}
    for r in kw:
        if r["chunk_id"] in merged:
            # If chunk already in semantic results, add BM25 score
            merged[r["chunk_id"]]["bm25"] = r.get("bm25", 0.0)
        else:
            # If only in BM25 results, add with default semantic score
            r["score"] = 0.0
            merged[r["chunk_id"]] = r

    # Convert merged dict to list
    out = list(merged.values())

    # Sort results by semantic score first, then BM25 score
    out.sort(key=lambda x: (x.get("score", 0.0), x.get("bm25", 0.0)), reverse=True)

    # Return top-k results
    return out[:k]

# Run example query if script is executed directly
if __name__ == "__main__":
    q = "How do I generate CNES report as XLSX from SonarQube?"   # Example query
    results = hybrid_search(q, k=8)   # Perform hybrid search
    print(json.dumps(results, indent=2))   # Print results in pretty JSON format
