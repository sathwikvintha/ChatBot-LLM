import json
from pathlib import Path
from transformers import AutoTokenizer
import uuid

CLEAN_DIR = Path("data/parsed_clean")
CHUNK_DIR = Path("data/chunks")
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
MAX_TOKENS = 800
OVERLAP = 150

def chunk_text(text):
  tokens = tokenizer.encode(text, add_special_tokens=False)
  chunks = []
  start = 0
  while start < len(tokens):
    end = min(start + MAX_TOKENS, len(tokens))
    piece = tokenizer.decode(tokens[start:end])
    chunks.append(piece)
    if end == len(tokens): break
    start = end - OVERLAP
  return chunks

def main():
  for jf in CLEAN_DIR.glob("*.json"):
    items = json.loads(jf.read_text(encoding="utf-8"))
    out_chunks = []
    for it in items:
      parts = chunk_text(it["text"])
      for idx, part in enumerate(parts):
        out_chunks.append({
          "chunk_id": uuid.uuid4().hex,
          "doc_id": it["doc_id"],
          "source_path": it["source_path"],
          "page_number": it.get("page_number"),
          "section_heading": it.get("section_heading"),
          "doc_type": it.get("doc_type"),
          "text": part
        })
    (CHUNK_DIR / jf.name).write_text(json.dumps(out_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
  print("Chunks in data/chunks/")

if __name__ == "__main__":
  main()
