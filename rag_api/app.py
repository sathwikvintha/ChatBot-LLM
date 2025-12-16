# Import FastAPI and supporting libraries
from fastapi import FastAPI
from pydantic import BaseModel   # For request validation
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer  # For embeddings
import faiss                                           # Vector index
from transformers import AutoModelForCausalLM, AutoTokenizer  # Local LLM
import torch                                           # For GPU/CPU device handling

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
metas = json.loads((INDEX_DIR / "meta.json").read_text(encoding="utf-8"))  # Load metadata
index = faiss.read_index(str(INDEX_DIR / "index.faiss"))                   # Load FAISS index
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2") # Embedding model

# -----------------------------
# Load local LLM (placeholder)
# -----------------------------
# Here we use Microsoft Phi-3-mini as a lightweight local model.
# Replace with your preferred model or endpoint (e.g., Ollama, OpenAI API).
llm_name = "microsoft/Phi-3-mini-4k-instruct"
tok = AutoTokenizer.from_pretrained(llm_name)          # Tokenizer
llm = AutoModelForCausalLM.from_pretrained(llm_name)   # Model
device = "cuda" if torch.cuda.is_available() else "cpu" # Use GPU if available
llm.to(device)

# -----------------------------
# Request schema
# -----------------------------
class Query(BaseModel):
    question: str   # User's question
    top_k: int = 8  # Number of chunks to retrieve (default 8)

# -----------------------------
# Retrieval function
# -----------------------------
def retrieve(question, k=8):
    """
    Encode the question into an embedding and search FAISS index.
    Returns top-k matching chunks with scores.
    """
    q = embed_model.encode([question], normalize_embeddings=True)
    D, I = index.search(q, k)   # D = distances, I = indices
    results = []
    for score, idx in zip(D[0], I[0]):
        m = metas[idx]
        results.append({**m, "score": float(score)})
    return results

# -----------------------------
# Build context for LLM prompt
# -----------------------------
def build_context(chunks):
    """
    Build a context string from retrieved chunks.
    Includes source info and chunk text.
    """
    blocks = []
    for c in chunks:
        header = f"Source: {c['source_path']} | Page: {c.get('page_number')} | Section: {c.get('section_heading')}"
        text = get_chunk_text(c["chunk_id"])
        blocks.append(header + "\n" + text)
    return "\n\n---\n\n".join(blocks)

# -----------------------------
# Helper: load chunk text
# -----------------------------
def get_chunk_text(chunk_id):
    """
    Scan chunk JSON files to find text for a given chunk_id.
    (Simple implementation; can be optimized with indexing.)
    """
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
    """
    Chat endpoint:
    - Retrieve relevant chunks
    - Build context
    - Generate answer using local LLM
    - Return answer + citations
    """
    chunks = retrieve(q.question, k=q.top_k)
    if not chunks:
        return {"answer": "I couldn't find relevant context in the documents.", "citations": []}

    # Build context from top chunks
    context = build_context(chunks[:6])

    # Construct prompt for LLM
    prompt = (
        "System: Answer only using the provided context. If context is insufficient, say so. Cite sources.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {q.question}\n"
        "Answer:"
    )

    # Tokenize and run through LLM
    inputs = tok(prompt, return_tensors="pt").to(device)
    out = llm.generate(**inputs, max_new_tokens=300, do_sample=False)

    # Decode output and extract answer
    answer = tok.decode(out[0], skip_special_tokens=True).split("Answer:")[-1].strip()

    # Build citations from retrieved chunks
    citations = [{
        "source_path": c["source_path"],
        "page_number": c.get("page_number"),
        "section_heading": c.get("section_heading"),
        "chunk_id": c["chunk_id"]
    } for c in chunks[:6]]

    return {"answer": answer, "citations": citations}

# -----------------------------
# Run server
# -----------------------------
# Command to run:
# uvicorn rag_api.app:app --reload
