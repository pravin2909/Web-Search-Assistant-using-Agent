#Document & Web Question Answering System (RAG + Agent)

This project is a Flask-based AI application that allows users to ask questions using:

Uploaded PDF documents (Document Mode)

Live web search results (Web Mode)

It uses Retrieval-Augmented Generation (RAG) combined with an LLM agent to produce accurate, context-aware answers.

**Features**

Upload PDF documents and ask questions from them

Real-time web search using DuckDuckGo

FAISS vector store for fast similarity search

Conversational memory for multi-turn chat

Powered by Groq LLaMA-3 (70B)

Simple Flask web interface

**System Architecture**

PDF Ingestion

PDFs are read using PyPDF2

Text is split into chunks using RecursiveCharacterTextSplitter

Chunks are embedded using sentence-transformers/all-MiniLM-L6-v2

Stored in FAISS vector database

Question Answering

Document Mode: answers are generated from PDF context

Web Mode: answers are generated from live web content

Relevant context is passed to the LLM agent

LLM Agent

Uses Groq LLaMA-3-70B

Integrated with DuckDuckGo Search tool

Maintains chat history using ConversationBufferMemory

**Tech Stack**

Backend: Flask

LLM: Groq LLaMA-3-70B

Embeddings: HuggingFace Sentence Transformers

Vector Database: FAISS

Agent Framework: LangChain

Web Search: DuckDuckGo

PDF Processing: PyPDF2

Web Scraping: BeautifulSoup

**Installation**
Clone the Repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

Create Virtual Environment
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows

**Install Dependencies**
pip install -r requirements.txt

Environment Variables

Create a .env file in the root directory:

GROQ_API_KEY=your_groq_api_key_here

Run the Application
python app.py


Access the application at:

http://localhost:5000

How to Use

Upload a PDF file

Select mode:

Document: query the uploaded PDF

Web: query live web content

Enter your question

Receive an AI-generated response

Project Structure
├── app.py
├── templates/
│   └── index.html
├── uploads/
├── .env
├── requirements.txt
└── README.md

**Use Cases**

Document-based question answering

Research and study assistance

Knowledge retrieval systems

Interview preparation tools

Future Improvements

Source citation and highlighting

Streaming responses

Multi-document support

Authentication and session management

Production deployment using Docker

**Author**
Pravin S
AI and Machine Learning Engineer
Email: pravinselvaraj077@gmail.com
Real-Time Web Search: Integrates DuckDuckGo Search for up-to-date information.

Conversation Memory: Remembers past interactions to enhance user experience.

Customizable Settings: Allows users to select AI models, adjust creativity levels, and manage conversation memory.

User-Friendly Interface: Designed with an intuitive chat-based UI using Streamlit.

**Prerequisites**

Ensure you have the following installed:

Python 3.8+

Streamlit

langchain
