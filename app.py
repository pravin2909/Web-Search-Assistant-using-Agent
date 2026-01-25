import os
from flask import Flask, request, render_template
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent
from langchain_community.tools import Tool, DuckDuckGoSearchResults
from langchain.memory import ConversationBufferMemory
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set")

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400
SIMILARITY_THRESHOLD = 0.5

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

vector_store = None

llm = ChatGroq(
    model_name="llama3-70b-8192",
    temperature=0.3,
    groq_api_key=GROQ_API_KEY
)

ddg = DuckDuckGoSearchResults()

ddg_tool = Tool(
    name="DuckDuckGo Search",
    func=ddg.run,
    description="Search the web for current information"
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="output"
)

agent = initialize_agent(
    tools=[ddg_tool],
    llm=llm,
    agent="conversational-react-description",
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)

def ingest_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = text_splitter.split_text(text)
        return FAISS.from_texts(chunks, embedding=embeddings)
    except Exception:
        return None

def web_scrape_content(query):
    try:
        result = ddg.run(query)
        if not result or not isinstance(result, list):
            return ""
        url = result[0].get("link", "")
        if not url:
            return ""
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
        return "\n".join(paragraphs)
    except Exception:
        return ""

def answer_question(question, mode):
    global vector_store
    try:
        if mode == "Document" and vector_store:
            docs = vector_store.similarity_search(question, k=3)
            if docs:
                context = "\n\n".join(d.page_content for d in docs)
                return agent.run(
                    input=f"Context from document:\n{context}\n\nQuestion: {question}"
                )
        elif mode == "Web":
            content = web_scrape_content(question)
            if content:
                return agent.run(
                    input=f"Web search results:\n{content}\n\nQuestion: {question}"
                )
        return agent.run(input=question)
    except Exception:
        return "Sorry, I encountered an error processing your request."

chat_history = []

@app.route("/", methods=["GET", "POST"])
def index():
    global vector_store
    if request.method == "POST":
        mode = request.form.get("mode", "Document")
        question = request.form.get("question", "").strip()
        pdf_file = request.files.get("pdf_file")

        if pdf_file and pdf_file.filename:
            if not pdf_file.filename.lower().endswith(".pdf"):
                return render_template(
                    "index.html",
                    error="Please upload a PDF file",
                    chat_history=chat_history
                )
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_file.filename)
            pdf_file.save(file_path)
            vector_store = ingest_pdf(file_path)

        if question:
            response = answer_question(question, mode)
            chat_history.append(("You", question))
            chat_history.append(("Assistant", response))

    return render_template("index.html", chat_history=chat_history)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
