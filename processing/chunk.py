# Import standard libraries
import json              # For reading/writing JSON files
from pathlib import Path # For handling file paths in a clean, cross-platform way
import uuid              # For generating unique IDs for chunks

# Import third-party libraries
from transformers import AutoTokenizer  # HuggingFace tokenizer for splitting text into tokens

# -----------------------------
# Define directories
# -----------------------------
CLEAN_DIR = Path("data/parsed_clean")   # Directory containing cleaned JSON documents
CHUNK_DIR = Path("data/chunks")         # Output directory for chunked JSON files
CHUNK_DIR.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

# -----------------------------
# Tokenizer setup
# -----------------------------
# Using BERT tokenizer to split text into tokens
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Parameters for chunking
MAX_TOKENS = 800   # Maximum tokens per chunk
OVERLAP = 150      # Overlap between consecutive chunks

# -----------------------------
# Function: chunk_text
# -----------------------------
def chunk_text(text):
    """
    Split text into overlapping chunks based on token count.
    - MAX_TOKENS: maximum tokens per chunk
    - OVERLAP: number of tokens overlapping between consecutive chunks
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)  # Convert text to token IDs
    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + MAX_TOKENS, len(tokens))             # End index for chunk
        piece = tokenizer.decode(tokens[start:end])            # Decode tokens back to text
        chunks.append(piece)

        if end == len(tokens):  # Stop if we've reached the end
            break

        # Move start forward but keep overlap
        start = end - OVERLAP

    return chunks

# -----------------------------
# Function: main
# -----------------------------
def main():
    """
    Main pipeline:
    - Iterate through all JSON files in CLEAN_DIR
    - For each document, split text into chunks
    - Build metadata for each chunk
    - Save chunks into CHUNK_DIR
    """
    for jf in CLEAN_DIR.glob("*.json"):
        items = json.loads(jf.read_text(encoding="utf-8"))  # Load JSON file
        out_chunks = []

        # Process each item in the JSON file
        for it in items:
            parts = chunk_text(it["text"])  # Split text into chunks
            for idx, part in enumerate(parts):
                out_chunks.append({
                    "chunk_id": uuid.uuid4().hex,      # Unique ID for chunk
                    "doc_id": it["doc_id"],            # Document ID
                    "source_path": it["source_path"],  # Original file path
                    "page_number": it.get("page_number"),       # Optional metadata
                    "section_heading": it.get("section_heading"), # Optional metadata
                    "doc_type": it.get("doc_type"),    # Document type
                    "text": part                       # Chunk text
                })

        # Save all chunks for this document into CHUNK_DIR
        (CHUNK_DIR / jf.name).write_text(
            json.dumps(out_chunks, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    print("✅ Chunks saved in data/chunks/")

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    main()
