import re, json
from pathlib import Path
from uuid import uuid4

RAW_DIR = Path("data/source")       # Folder with raw .txt or .pdf files
CHUNK_DIR = Path("data/chunks")         # Output folder for chunked JSON
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_text(text: str, chunk_size=1200, overlap=200):
    text = clean_text(text)
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks

def chunk_document(path: Path):
    doc_id = path.stem
    raw_text = path.read_text(encoding="utf-8")
    chunks = split_text(raw_text)
    chunk_items = []

    for i, chunk in enumerate(chunks):
        chunk_items.append({
            "chunk_id": f"{doc_id}_{i}",
            "doc_id": doc_id,
            "source_path": str(path),
            "page_number": None,  # Optional: fill if you parse PDFs
            "section_heading": None,  # Optional: fill if you detect headings
            "doc_type": "text",
            "text": chunk
        })

    out_path = CHUNK_DIR / f"{doc_id}.json"
    out_path.write_text(json.dumps(chunk_items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Chunked {doc_id}: {len(chunk_items)} chunks")

def main():
    for file in RAW_DIR.glob("*.txt"):
        chunk_document(file)

if __name__ == "__main__":
    main()
