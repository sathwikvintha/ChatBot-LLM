# Import standard libraries
import re                # For regular expressions (used to clean text)
import json              # For reading/writing JSON files
from pathlib import Path # For handling file paths in a clean, cross-platform way
from uuid import uuid4   # For generating unique IDs (not used here, but imported)

# -----------------------------
# Define directories
# -----------------------------
RAW_DIR = Path("data/source")       # Folder containing raw input files (.txt or .pdf)
CHUNK_DIR = Path("data/chunks")     # Output folder where chunked JSON files will be saved
CHUNK_DIR.mkdir(parents=True, exist_ok=True)  # Ensure output folder exists

# -----------------------------
# Utility functions
# -----------------------------

def clean_text(text: str) -> str:
    """
    Clean text by:
    - Replacing multiple whitespace characters with a single space
    - Stripping leading/trailing spaces
    """
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_text(text: str, chunk_size=1200, overlap=200):
    """
    Split text into overlapping chunks.
    - chunk_size: number of characters per chunk
    - overlap: number of characters overlapping between consecutive chunks
    """
    text = clean_text(text)   # Clean text before splitting
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))   # End index for chunk
        chunk = text[start:end]                    # Slice text
        chunks.append(chunk)

        if end == len(text):   # Stop if we've reached the end of text
            break

        # Move start forward but keep overlap
        start = end - overlap

    return chunks


def chunk_document(path: Path):
    """
    Process a single document:
    - Read raw text
    - Split into chunks
    - Build metadata for each chunk
    - Save chunks as JSON file
    """
    doc_id = path.stem   # Use filename (without extension) as document ID
    raw_text = path.read_text(encoding="utf-8")   # Read file content
    chunks = split_text(raw_text)                 # Split into chunks
    chunk_items = []

    # Build metadata for each chunk
    for i, chunk in enumerate(chunks):
        chunk_items.append({
            "chunk_id": f"{doc_id}_{i}",    # Unique ID per chunk
            "doc_id": doc_id,               # Document ID
            "source_path": str(path),       # Original file path
            "page_number": None,            # Optional: fill if parsing PDFs
            "section_heading": None,        # Optional: fill if detecting headings
            "doc_type": "text",             # Type of document
            "text": chunk                   # Actual chunk text
        })

    # Save chunks to JSON file
    out_path = CHUNK_DIR / f"{doc_id}.json"
    out_path.write_text(
        json.dumps(chunk_items, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"Chunked {doc_id}: {len(chunk_items)} chunks")


def main():
    """
    Main entry point:
    - Iterate through all .txt files in RAW_DIR
    - Chunk each document
    """
    for file in RAW_DIR.glob("*.txt"):
        chunk_document(file)


# -----------------------------
# Run script
# -----------------------------
if __name__ == "__main__":
    main()
