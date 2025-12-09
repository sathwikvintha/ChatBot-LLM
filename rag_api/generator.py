# rag_api/generator.py

import os
from typing import List, Dict
from openai import OpenAI

# Load API key from environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def build_prompt(question: str, contexts: List[Dict]) -> str:
    """
    Build a prompt with retrieved contexts and the user question.
    """
    context_blocks = []
    for i, c in enumerate(contexts, start=1):
        label = f"[{i}] {c.get('source_path','unknown')}"
        section = c.get("section_heading")
        page = c.get("page_number")
        meta = f"{label}" + (f" — {section}" if section else "") + (f" — Page {page}" if page else "")
        text = c.get("text", "").strip()
        context_blocks.append(f"{meta}\n{text}")

    context_text = "\n\n".join(context_blocks[:6])  # limit to top 6 chunks

    return (
        "You are a helpful assistant. Answer the user's question using ONLY the context below. "
        "Be concise, structured, and include citation markers like [1], [2] "
        "corresponding to the sources used.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

def _extractive_fallback_answer(question: str, contexts: List[Dict]) -> str:
    """
    Build a concise, extractive answer directly from contexts when LLM is unavailable.
    Produces short bullets and maps them to sources [1], [2], ...
    """
    if not contexts:
        return "No relevant context available to answer this question."

    bullets = []
    # Use up to 4 top contexts to keep the answer tight
    for i, c in enumerate(contexts[:4], start=1):
        text = (c.get("text") or "").strip()
        if not text:
            continue
        # Heuristic: take the most informative 1–2 sentences
        parts = [p.strip() for p in text.split(".") if p.strip()]
        summary = ". ".join(parts[:2])
        source = c.get("source_path", "unknown")
        section = c.get("section_heading")
        page = c.get("page_number")
        meta = f"[{i}] {source}" + (f" — {section}" if section else "") + (f" — Page {page}" if page else "")
        bullets.append(f"- {summary} [{i}]")

    if not bullets:
        return "Context retrieved, but no extractive sentences found."

    return (
        "Answer (extractive):\n"
        + "\n".join(bullets)
        + "\n\nCitations:\n"
        + "\n".join([f"[{i+1}] {contexts[i].get('source_path','unknown')}" for i in range(min(4, len(contexts)))])
    )

def generate_answer(question: str, contexts: List[Dict]) -> str:
    """
    Generate an answer using OpenAI GPT model with graceful fallback.
    """
    prompt = build_prompt(question, contexts)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Fallback: produce a concise extractive answer from contexts
        return _extractive_fallback_answer(question, contexts)
