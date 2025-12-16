###### **ChatBot-LLM**



An enterprise-ready chatbot that combines a FastAPI backend with an Angular frontend to enable document ingestion, semantic search, and conversational interfaces.





**Project Structure**



* **chatbot/ ->** Python backend (FastAPI, embeddings, retrieval)
* **org-docs-chatbot/** -> Angular frontend (UI for chat, document upload/search)





**Prerequisites**



* **Python 3.10+** installed and on PATH
* **Node.js  18+** and **NPM** installed
* **Angular CLI** (npm install -g @angular/cli)





**Clone the repository**



git clone https://github.com/sathwikvintha/ChatBot-LLM.git

cd ChatBot-LLM







###### **Backend setup (FastAPI)**



1. **Create and activate a virtual environment**



&nbsp;	cd chatbot

&nbsp;	python -m venv .venv

&nbsp;	.\\.venv\\Scripts\\Activate.ps1



2\. **Install dependencies**	



	pip install -r requirements.txt



3\. **Environment variables**

	

	EMBEDDINGS\_MODEL=text-embedding-3-large

&nbsp;	VECTOR\_INDEX\_PATH=./data/index

&nbsp;	DOCS\_DIR=../org-docs-chatbot/assets/docs

&nbsp;	ALLOW\_CORS\_ORIGINS=http://localhost:4200



4\. **Run the FastAPI server**



	uvicorn main:app --reload

&nbsp;	Default - http://127.0.0.1:8000







###### **Frontend setup (Angular)**



1. **Install dependencies**



	cd ../org-docs-chatbot

&nbsp;	npm install



2\. **Run the Angular dev server**



	ng serve





###### **Data ingestion and search**



* Place documents inside chatbot/data/source
* Run the ingestion script (python ingest.py)







###### **Quick start**



1. **Backend:**



&nbsp;	cd chatbot

&nbsp;	python -m venv .venv \&\& .\\.venv\\Scripts\\Activate.ps1

&nbsp;	pip install -r requirements.txt

&nbsp;	uvicorn main:app --reload



2\. **Frontend:**



&nbsp;	cd ../org-docs-chatbot

&nbsp;	npm install

&nbsp;	ng serve



3\. Open http://localhost:4200 and use the app.



