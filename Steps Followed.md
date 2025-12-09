1\. Created Project Structure



Structure:

ingestion/

processing/

indexing/

retriever/

rag\_api/

ui/

data/raw\_docs/

data/parsed/

data/chunks/

data/faiss\_index/





2\. Installed dependencies



**python -m venv .venv**

**source .venv\\Scripts\\activate**

**python.exe -m pip install --upgrade pip**



**pip install pdfplumber python-docx pymupdf pytesseract pillow opencv-python**  (pdfplumber - Extracts text from PDFs page by page, python-docx - Reads Word (.docx) files and extracts headings/paragraphs, pymupdf - Alternative PDF parser; also lets us render pages as images for OCR fallback, pytesseract - OCR engine to read text from scanned PDFs or images, pillow - Image processing library (needed by Tesseract), Opencv-python - Extra image utilities (preprocessing before OCR, e.g., grayscale, thresholding))



**pip install sentence-transformers faiss-cpu numpy pandas tqdm**  (sentence-transformers - Provides models like all-MiniLM-L6-v2 to convert text chunks into embeddings (vectors), faiss-cpu - Facebook AI Similarity Search; stores embeddings and performs fast similarity search, numpy - Core numerical library for handling embeddings and arrays, pandas - For structured data manipulation (metadata, chunk management), tqdm - Progress bars during ingestion/embedding)



**pip install rank\_bm25 transformers torch** (rank\_bm25 - Keyword-based search (BM25 algorithm) to complement semantic search for acronyms/IDs, transformers - Hugging Face library for LLMs and tokenizers (used for chunking, reranking, and response generation), torch - Backend for running transformer models locally)



**pip install fastapi uvicorn streamlit (fastapi** - Backend framework to expose endpoints (/search, /chat), uvicorn - ASGI server to run FastAPI apps, streamlit - Quick web UI for testing queries and displaying answers + citations)





3\. run 'python ingest.py' 

