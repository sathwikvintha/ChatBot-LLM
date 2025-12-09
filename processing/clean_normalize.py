import json, re, hashlib
from pathlib import Path

PARSED_DIR = Path("data/parsed")
CLEAN_DIR = Path("data/parsed_clean")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

FOOTER_PATTERNS = [
  r"Confidential- Oracle Internal"
]

def normalize_text(t):
  t = t.replace("\r", "\n")
  t = re.sub(r"[ \t]+", " ", t)
  t = re.sub(r"\n{2,}", "\n", t)
  t = t.strip()
  return t

def is_footer(t):
  return any(re.search(p, t, flags=re.I) for p in FOOTER_PATTERNS)

def main():
  for jf in PARSED_DIR.glob("*.json"):
    items = json.loads(jf.read_text(encoding="utf-8"))
    seen = set()
    cleaned = []
    for it in items:
      txt = normalize_text(it["text"])
      if is_footer(txt): continue
      h = hashlib.md5(txt.encode("utf-8")).hexdigest()
      key = (it["doc_id"], it.get("page_number"), it.get("section_heading"), h)
      if key in seen: continue
      seen.add(key)
      it["text"] = txt
      cleaned.append(it)
    out = CLEAN_DIR / jf.name
    out.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
  print("Cleaned artifacts in data/parsed_clean/")

if __name__ == "__main__":
  main()
