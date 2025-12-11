from openai import OpenAI

# Connect to local Ollama server
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="not-needed"
)

def generate_answer(question, retrieved_chunks):
    # Combine retrieved context
    context = "\n\n".join([chunk.get("text", "") for chunk in retrieved_chunks])

    prompt = f"""
    You are a helpful assistant. Use ONLY the context below to answer the question.
    Do NOT copy text directly; summarize clearly.

    Context:
    {context}

    Question: {question}

    Answer:
    """

    # Call local LLM
    response = client.chat.completions.create(
        model="mistral",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )

    # Extract answer correctly
    answer = response.choices[0].message.content

    # Build SAFE citations
    citations = []
    for chunk in retrieved_chunks[:3]:
        meta = chunk.get("meta", {})   # safe fallback
        snippet = chunk.get("text", "")[:200]
        citations.append({
            "source": meta.get("source", "unknown"),
            "snippet": snippet
        })

    return answer, citations
