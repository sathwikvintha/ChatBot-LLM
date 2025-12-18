# app.py

import json, time, logging
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel   # For request validation
from sentence_transformers import SentenceTransformer  # For embeddings
import faiss                                           # Vector index
from transformers import AutoModelForCausalLM, AutoTokenizer  # Local LLM
import torch                                           # For GPU/CPU device handling

# -----------------------------
# Setup logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# -----------------------------
# Initialize FastAPI app
# -----------------------------
app = FastAPI()

# -----------------------------
# Define directories
# -----------------------------
INDEX_DIR = Path("data/faiss_index")   # Directory with FAISS index + metadata
CHUNK_DIR = Path("data/chunks")        # Directory with chunked documents

# -----------------------------
# Load FAISS index and metadata
# -----------------------------
t0 = time.time()
logging.info("Loading FAISS index and metadata...")
metas = json.loads((INDEX_DIR / "meta.json").read_text(encoding="utf-8"))
index = faiss.read_index(str(INDEX_DIR / "index.faiss"))
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
logging.info(f"FAISS + embeddings loaded in {time.time() - t0:.2f}s")

# -----------------------------
# Load local LLM (placeholder)
# -----------------------------
t1 = time.time()
logging.info("Loading local LLM...")
llm_name = "microsoft/Phi-3-mini-4k-instruct"
tok = AutoTokenizer.from_pretrained(llm_name)
llm = AutoModelForCausalLM.from_pretrained(llm_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
llm.to(device)
logging.info(f"LLM loaded in {time.time() - t1:.2f}s")

# -----------------------------
# Request schema
# -----------------------------
class Query(BaseModel):
    question: str
    top_k: int = 8

# -----------------------------
# Retrieval function
# -----------------------------
def retrieve(question, k=8):
    t_start = time.time()
    q = embed_model.encode([question], normalize_embeddings=True)
    logging.info(f"Query encoded in {time.time() - t_start:.2f}s")

    D, I = index.search(q, k)
    logging.info(f"FAISS search completed in {time.time() - t_start:.2f}s")

    results = []
    for score, idx in zip(D[0], I[0]):
        m = metas[idx]
        results.append({**m, "score": float(score)})

    logging.info(f"Retrieved {len(results)} chunks in {time.time() - t_start:.2f}s total")
    return results

# -----------------------------
# Build context for LLM prompt
# -----------------------------
def build_context(chunks):
    t_start = time.time()
    blocks = []
    for c in chunks:
        header = f"Source: {c['source_path']} | Page: {c.get('page_number')} | Section: {c.get('section_heading')}"
        text = get_chunk_text(c["chunk_id"])
        blocks.append(header + "\n" + text)
    logging.info(f"Built context in {time.time() - t_start:.2f}s")
    return "\n\n---\n\n".join(blocks)

# -----------------------------
# Helper: load chunk text
# -----------------------------
def get_chunk_text(chunk_id):
    for jf in CHUNK_DIR.glob("*.json"):
        items = json.loads(jf.read_text(encoding="utf-8"))
        for it in items:
            if it["chunk_id"] == chunk_id:
                return it["text"]
    return ""

# -----------------------------
# FastAPI endpoint: /chat
# -----------------------------
@app.post("/chat")
def chat(q: Query):
    t_total = time.time()
    logging.info("Received /chat request")

    # Step 1: Retrieval
    chunks = retrieve(q.question, k=q.top_k)
    if not chunks:
        logging.warning("No chunks retrieved")
        return {"answer": "I couldn't find relevant context in the documents.", "citations": []}

    # Step 2: Context building
    context = build_context(chunks[:6])

    # Step 3: Prompt construction
    t_prompt = time.time()
    prompt = (
        "System: Answer only using the provided context. If context is insufficient, say so. Cite sources.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {q.question}\n"
        "Answer:"
    )
    logging.info(f"Prompt built in {time.time() - t_prompt:.2f}s")

    # Step 4: LLM generation
    t_llm = time.time()
    inputs = tok(prompt, return_tensors="pt").to(device)
    out = llm.generate(**inputs, max_new_tokens=300, do_sample=False)
    answer = tok.decode(out[0], skip_special_tokens=True).split("Answer:")[-1].strip()
    logging.info(f"LLM generation completed in {time.time() - t_llm:.2f}s")

    # Step 5: Build citations
    citations = [{
        "source_path": c["source_path"],
        "page_number": c.get("page_number"),
        "section_heading": c.get("section_heading"),
        "chunk_id": c["chunk_id"]
    } for c in chunks[:6]]

    logging.info(f"Total /chat request processed in {time.time() - t_total:.2f}s")

    return {"answer": answer, "citations": citations}
