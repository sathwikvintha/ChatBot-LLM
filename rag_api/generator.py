# Import the OpenAI client library
from openai import OpenAI

# -----------------------------------
# Connect to local Ollama server
# -----------------------------------
# We initialize the OpenAI client but point it to a local Ollama server
# running on port 11434. Ollama exposes an OpenAI-compatible API.
# The api_key is not needed for local use, but the parameter must be set.
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="not-needed"
)

# -----------------------------------
# Function: generate_answer
# -----------------------------------
def generate_answer(question, retrieved_chunks):
    """
    Generate an answer to a given question using retrieved context chunks.
    - question: the user’s query
    - retrieved_chunks: list of chunks (dicts) containing text and metadata
    """

    # Combine retrieved context into a single string
    # Each chunk’s text is joined with double newlines for readability
    context = "\n\n".join([chunk.get("text", "") for chunk in retrieved_chunks])

    # Build the prompt for the LLM
    # The assistant is instructed to use ONLY the provided context
    # and to summarize clearly instead of copying text verbatim.
    prompt = f"""
    You are a helpful assistant. Use ONLY the context below to answer the question.
    Do NOT copy text directly; summarize clearly.

    Context:
    {context}

    Question: {question}

    Answer:
    """

    # -----------------------------------
    # Call local LLM via Ollama
    # -----------------------------------
    # We send the prompt to the local Ollama server using the "mistral" model.
    # The request is structured as a chat completion with a single user message.
    response = client.chat.completions.create(
        model="mistral",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300   # Limit the response length
    )

    # Extract the answer text from the response object
    answer = response.choices[0].message.content

    # -----------------------------------
    # Build SAFE citations
    # -----------------------------------
    # To provide transparency, we include snippets from the retrieved chunks.
    # Only the first 3 chunks are used for citations.
    citations = []
    for chunk in retrieved_chunks[:3]:
        # Get metadata safely (fallback to empty dict if missing)
        meta = chunk.get("meta", {})
        # Take the first 200 characters of the chunk text as a snippet
        snippet = chunk.get("text", "")[:200]
        citations.append({
            "source": meta.get("source", "unknown"),  # Source file or identifier
            "snippet": snippet                        # Short preview of the text
        })

    # Return both the generated answer and the citations
    return answer, citations
