# Import standard libraries
import json              # For reading/writing JSON files
import re                # For regular expressions (used to clean text and detect footers)
import hashlib           # For generating hashes to detect duplicates
from pathlib import Path # For handling file paths in a clean, cross-platform way

# -----------------------------
# Define directories
# -----------------------------
PARSED_DIR = Path("data/parsed")        # Directory containing parsed JSON documents
CLEAN_DIR = Path("data/parsed_clean")   # Output directory for cleaned JSON documents
CLEAN_DIR.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

# -----------------------------
# Footer patterns to remove
# -----------------------------
# Any text matching these patterns will be considered a footer and removed
FOOTER_PATTERNS = [
    r"Confidential- Oracle Internal"
]

# -----------------------------
# Function: normalize_text
# -----------------------------
def normalize_text(t: str) -> str:
    """
    Normalize text by:
    - Converting carriage returns (\r) to newlines (\n)
    - Replacing multiple spaces/tabs with a single space
    - Collapsing multiple newlines into one
    - Stripping leading/trailing whitespace
    """
    t = t.replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    t = t.strip()
    return t

# -----------------------------
# Function: is_footer
# -----------------------------
def is_footer(t: str) -> bool:
    """
    Check if text matches any footer pattern.
    Returns True if text is a footer, False otherwise.
    """
    return any(re.search(p, t, flags=re.I) for p in FOOTER_PATTERNS)

# -----------------------------
# Main cleaning pipeline
# -----------------------------
def main():
    # Iterate through all parsed JSON files
    for jf in PARSED_DIR.glob("*.json"):
        items = json.loads(jf.read_text(encoding="utf-8"))  # Load JSON file
        seen = set()     # Track unique entries to avoid duplicates
        cleaned = []     # Store cleaned items

        # Process each item in the JSON file
        for it in items:
            txt = normalize_text(it["text"])   # Normalize text

            if is_footer(txt):                 # Skip if text is a footer
                continue

            # Generate hash of text for deduplication
            h = hashlib.md5(txt.encode("utf-8")).hexdigest()

            # Build a unique key combining doc_id, page_number, section_heading, and text hash
            key = (it["doc_id"], it.get("page_number"), it.get("section_heading"), h)

            if key in seen:                    # Skip if duplicate
                continue

            seen.add(key)                      # Mark as seen
            it["text"] = txt                   # Replace text with cleaned version
            cleaned.append(it)                 # Add to cleaned list

        # Save cleaned items to CLEAN_DIR
        out = CLEAN_DIR / jf.name
        out.write_text(
            json.dumps(cleaned, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    print("✅ Cleaned artifacts saved in data/parsed_clean/")

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    main()
