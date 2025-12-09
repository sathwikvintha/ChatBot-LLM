import json, numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

CHUNK_DIR = Path("data/chunks")
INDEX_DIR = Path("data/faiss_index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def main():
    texts, metas = [], []

    for jf in CHUNK_DIR.glob("*.json"):
        items = json.loads(jf.read_text(encoding="utf-8"))
        for it in items:
            texts.append(it["text"])
            metas.append({
                "chunk_id": it["chunk_id"],
                "doc_id": it["doc_id"],
                "source_path": it["source_path"],
                "page_number": it.get("page_number"),
                "section_heading": it.get("section_heading"),
                "doc_type": it.get("doc_type")
            })

    emb = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    dim = emb.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 200
    index.add(emb)

    faiss.write_index(index, str(INDEX_DIR / "index.faiss"))
    (INDEX_DIR / "meta.json").write_text(json.dumps(metas, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Indexed {len(texts)} chunks.")

if __name__ == "__main__":
    main()
