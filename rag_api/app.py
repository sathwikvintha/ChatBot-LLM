from fastapi import FastAPI
from pydantic import BaseModel
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

INDEX_DIR = Path("data/faiss_index")
CHUNK_DIR = Path("data/chunks")

metas = json.loads((INDEX_DIR / "meta.json").read_text(encoding="utf-8"))
index = faiss.read_index(str(INDEX_DIR / "index.faiss"))
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Lightweight local LLM (placeholder). Replace with your preferred endpoint.
llm_name = "microsoft/Phi-3-mini-4k-instruct"
tok = AutoTokenizer.from_pretrained(llm_name)
llm = AutoModelForCausalLM.from_pretrained(llm_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
llm.to(device)

class Query(BaseModel):
  question: str
  top_k: int = 8

def retrieve(question, k=8):
  q = embed_model.encode([question], normalize_embeddings=True)
  D, I = index.search(q, k)
  results = []
  for score, idx in zip(D[0], I[0]):
    m = metas[idx]
    results.append({**m, "score": float(score)})
  return results

def build_context(chunks):
  blocks = []
  for c in chunks:
    header = f"Source: {c['source_path']} | Page: {c.get('page_number')} | Section: {c.get('section_heading')}"
    text = get_chunk_text(c["chunk_id"])
    blocks.append(header + "\n" + text)
  return "\n\n---\n\n".join(blocks)

# Helper: load full chunk text from data/chunks (simple scan; can be optimized)
def get_chunk_text(chunk_id):
  for jf in CHUNK_DIR.glob("*.json"):
    items = json.loads(jf.read_text(encoding="utf-8"))
    for it in items:
      if it["chunk_id"] == chunk_id:
        return it["text"]
  return ""

@app.post("/chat")
def chat(q: Query):
  chunks = retrieve(q.question, k=q.top_k)
  if not chunks:
    return {"answer": "I couldn't find relevant context in the documents.", "citations": []}
  context = build_context(chunks[:6])
  prompt = (
    "System: Answer only using the provided context. If context is insufficient, say so. Cite sources.\n\n"
    f"Context:\n{context}\n\n"
    f"Question: {q.question}\n"
    "Answer:"
  )
  inputs = tok(prompt, return_tensors="pt").to(device)
  out = llm.generate(**inputs, max_new_tokens=300, do_sample=False)
  answer = tok.decode(out[0], skip_special_tokens=True).split("Answer:")[-1].strip()

  citations = [{
    "source_path": c["source_path"],
    "page_number": c.get("page_number"),
    "section_heading": c.get("section_heading"),
    "chunk_id": c["chunk_id"]
  } for c in chunks[:6]]

  return {"answer": answer, "citations": citations}

# Run: uvicorn rag_api.app:app --reload
