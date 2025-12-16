# **ChatBot-LLM**

An enterprise-ready chatbot that combines a **FastAPI backend** with an **Angular frontend** to enable document ingestion, semantic search, and conversational interfaces.

---

## **📂 Project Structure**
- **chatbot/** → Python backend (FastAPI, embeddings, retrieval)
- **org-docs-chatbot/** → Angular frontend (UI for chat, document upload/search)

---

## **⚙️ Prerequisites**
- **Python 3.10+** installed and on PATH  
- **Node.js 18+** and **npm** installed  
- **Angular CLI**  
  ```bash
  npm install -g @angular/cli


## **🔗 Clone the Repository**
	git clone https://github.com/sathwikvintha/ChatBot-LLM.git
	cd ChatBot-LLM



## **🚀 Backend Setup (FastAPI)**
- Create and activate a virtual environment

		cd chatbot
		python -m venv .venv
		.\.venv\Scripts\Activate.ps1

- Install dependencies

		pip install -r requirements.txt

- Environment variables

  		EMBEDDINGS_MODEL=text-embedding-3-large
		VECTOR_INDEX_PATH=./data/index
		DOCS_DIR=../org-docs-chatbot/assets/docs
		ALLOW_CORS_ORIGINS=http://localhost:4200

- Run the FastAPI server

		Run the FastAPI server
  		Default URL → http://127.0.0.1:8000


## **🎨 Frontend Setup (Angular)**
- Install dependencies

  		cd ../org-docs-chatbot
		npm install

- Run the Angular dev server

  		ng serve
		Default URL → http://localhost:4200


## **📑 Data Ingestion & Search**
- Place documents inside:

		chatbot/data/source

- Run the ingestion script:

  		python ingest.py



## **⚡ Quick Start**
- Backend

		cd chatbot
		python -m venv .venv && .\.venv\Scripts\Activate.ps1
		pip install -r requirements.txt
		uvicorn main:app --reload


- Frontend

  		cd ../org-docs-chatbot
		npm install
		ng serve

- Open 👉 http://localhost:4200 and start using the app.



  






