# main.py

import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from rag_api.generator import generate_answer
from retriever.search import hybrid_search  # Uses robust meta loading internally

# ---------------------------
# FastAPI setup
# ---------------------------
app = FastAPI()

# Allow Angular frontend (localhost:4200) to call backend
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

    # Use hybrid search (semantic + keyword)
    results = hybrid_search(req.question, k=req.top_k)

    # Normalize contexts
    contexts = [{**r, "text": r.get("text", "")} for r in results]

    # ---------------------------
    # FIX: unpack the two values
    # ---------------------------
    answer_text, llm_citations = generate_answer(req.question, contexts)

    # Build frontend citations list
    citations = [
        Citation(
            source_path=c.get("source_path", "unknown"),
            page_number=c.get("page_number"),
            section_heading=c.get("section_heading"),
            snippet=c.get("text", "")[:200]
        )
        for c in contexts
    ]

    return ChatResponse(answer=answer_text, citations=citations)
