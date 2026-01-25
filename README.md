# Document & Web Question Answering System (RAG + Agent)

This project is a Flask-based AI application that allows users to ask questions using uploaded PDF documents or live web search. It combines Retrieval-Augmented Generation (RAG) with an LLM agent to generate accurate and context-aware responses.

---

## Features

- Upload PDF documents and ask questions from them  
- Real-time web search using DuckDuckGo  
- FAISS vector store for efficient similarity search  
- Conversational memory for multi-turn interactions  
- Powered by Groq LLaMA-3 (70B)  
- Simple Flask-based web interface  

---

## System Architecture

### 1. PDF Ingestion
- PDFs are read using PyPDF2  
- Text is split into chunks using RecursiveCharacterTextSplitter  
- Chunks are embedded using sentence-transformers/all-MiniLM-L6-v2  
- Embeddings are stored in FAISS  

### 2. Question Answering
- Document Mode: answers are generated from PDF context  
- Web Mode: answers are generated using live web content  
- Retrieved context is passed to the LLM agent  

### 3. LLM Agent
- Uses Groq LLaMA-3-70B  
- Integrated with DuckDuckGo Search tool  
- Maintains chat history using ConversationBufferMemory  

---

## Tech Stack

- Backend: Flask  
- LLM: Groq LLaMA-3-70B  
- Embeddings: HuggingFace Sentence Transformers  
- Vector Database: FAISS  
- Agent Framework: LangChain  
- Web Search: DuckDuckGo  
- PDF Processing: PyPDF2  
- Web Scraping: BeautifulSoup  

---

## Installation

### Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows

