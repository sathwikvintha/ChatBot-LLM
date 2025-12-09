# retriever/search.py

import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi

INDEX_DIR = Path("data/index")   # Align with ingest.py outputs
META_PATH = INDEX_DIR / "meta.json"
FAISS_PATH = INDEX_DIR / "faiss.index"

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index = faiss.read_index(str(FAISS_PATH))

def _load_metas(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    # meta.json can be a list or a dict keyed by chunk_id
    if isinstance(raw, dict):
        meta_list = list(raw.values())
    elif isinstance(raw, list):
        meta_list = raw
    else:
        raise ValueError("meta.json must be a list or a dict of chunk metadata")

    # Optional: ensure chunk_id exists in each item
    for i, m in enumerate(meta_list):
        if "chunk_id" not in m:
            m["chunk_id"] = m.get("id", f"chunk_{i}")
    return meta_list

metas = _load_metas(META_PATH)

# Basic validation: FAISS vectors count should match meta count (best effort)
try:
    ntotal = index.ntotal
    if ntotal != len(metas):
        print(f"⚠️ Warning: FAISS index count ({ntotal}) != meta count ({len(metas)}). "
              "Retrieval may misalign if ingest ordering differs.")
except Exception:
    pass

# Prepare BM25 corpus from meta texts
corpus_texts = [m.get("text", "") for m in metas]
tokenized_corpus = [t.lower().split() for t in corpus_texts]
bm25 = BM25Okapi(tokenized_corpus)

def faiss_topk(query: str, k: int = 10):
    q = model.encode([query], normalize_embeddings=True)
    D, I = index.search(q, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        m = metas[idx]
        results.append({**m, "score": float(score)})
    return results

def bm25_topk(query: str, k: int = 10):
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    idxs = np.argsort(scores)[-k:][::-1]
    results = []
    for i in idxs:
        m = metas[i]
        results.append({**m, "bm25": float(scores[i])})
    return results

def hybrid_search(query: str, k: int = 10):
    sem = faiss_topk(query, k)
    kw = bm25_topk(query, k)

    # Merge by chunk_id
    merged = {(r["chunk_id"]): r for r in sem}
    for r in kw:
        if r["chunk_id"] in merged:
            merged[r["chunk_id"]]["bm25"] = r.get("bm25", 0.0)
        else:
            r["score"] = 0.0
            merged[r["chunk_id"]] = r

    out = list(merged.values())
    out.sort(key=lambda x: (x.get("score", 0.0), x.get("bm25", 0.0)), reverse=True)
    return out[:k]

if __name__ == "__main__":
    q = "How do I generate CNES report as XLSX from SonarQube?"
    results = hybrid_search(q, k=8)
    print(json.dumps(results, indent=2))
