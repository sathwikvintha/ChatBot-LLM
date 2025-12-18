# main.py

import json, time, logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from rag_api.generator import generate_answer
from retriever.search import hybrid_search

# ---------------------------
# Setup logging
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ---------------------------
# FastAPI setup
# ---------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Request/Response models
# ---------------------------
class ChatRequest(BaseModel):
    question: str
    top_k: Optional[int] = 8

class Citation(BaseModel):
    source_path: str
    page_number: Optional[int] = None
    section_heading: Optional[str] = None
    snippet: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]

# ---------------------------
# Chat endpoint
# ---------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    t_total = time.time()
    logging.info("Received /chat request (main.py)")

    # Step 1: Hybrid search
    t_search = time.time()
    results = hybrid_search(req.question, k=req.top_k)
    logging.info(f"Hybrid search completed in {time.time() - t_search:.2f}s")

    # Step 2: Normalize contexts
    t_norm = time.time()
    contexts = [{**r, "text": r.get("text", "")} for r in results]
    logging.info(f"Contexts normalized in {time.time() - t_norm:.2f}s")

    # Step 3: Generate answer
    t_gen = time.time()
    answer_text, llm_citations = generate_answer(req.question, contexts)
    logging.info(f"Answer generated in {time.time() - t_gen:.2f}s")

    # Step 4: Build citations
    t_cite = time.time()
    citations = [
        Citation(
            source_path=c.get("source_path", "unknown"),
            page_number=c.get("page_number"),
            section_heading=c.get("section_heading"),
            snippet=c.get("text", "")[:200]
        )
        for c in contexts
    ]
    logging.info(f"Citations built in {time.time() - t_cite:.2f}s")

    logging.info(f"Total /chat request processed in {time.time() - t_total:.2f}s")

    return ChatResponse(answer=answer_text, citations=citations)
