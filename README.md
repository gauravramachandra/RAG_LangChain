# RAG Project

A comprehensive **Retrieval-Augmented Generation (RAG)** system built with FastAPI, Streamlit, LangChain, LangGraph, Pinecone, Groq, Tavily, and HuggingFace. This project combines document retrieval from a Pinecone vector database, real-time web search via Tavily, and advanced LLM reasoning (Groq Llama 3 70B) to deliver intelligent, context-aware responses through a modern web interface.

## 🚀 Features

- **Document Upload & Processing**: Upload PDF documents that are automatically processed and indexed
- **Intelligent Routing**: Smart agent that decides between RAG retrieval, web search, or direct answering
- **Vector Database**: Powered by Pinecone for efficient document storage and retrieval
- **Web Search Integration**: Real-time web search using Tavily for up-to-date information
- **FastAPI Backend**: RESTful API with async endpoints
- **Streamlit Frontend**: User-friendly web interface
- **Session Management**: Persistent conversation history
- **Configurable Web Search**: Toggle web search on/off per query

## 🛠️ Tech Stack

### Backend

- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern, fast web framework for building APIs
- **[LangChain](https://langchain.readthedocs.io/)** - Framework for developing applications with LLMs
- **[LangGraph](https://langchain-ai.github.io/langgraph/)** - Library for building stateful, multi-actor applications
- **[Pinecone](https://www.pinecone.io/)** - Vector database for similarity search
- **[Groq](https://groq.com/)** - Fast LLM inference (Llama 3 70B model)
- **[Tavily](https://tavily.com/)** - Web search API for real-time information
- **[HuggingFace](https://huggingface.co/)** - Embeddings model (sentence-transformers)

### Frontend

- **[Streamlit](https://streamlit.io/)** - Python web app framework
- **Session Management** - Persistent chat history
- **File Upload** - Drag-and-drop PDF processing

### AI Models & Tools

- **Llama 3 70B** (via Groq) - Large Language Model for reasoning and generation
- **sentence-transformers/all-MiniLM-L6-v2** - Embedding model for document vectorization
- **Tavily Search** - Real-time web search integration
- **PyPDF** - PDF document processing
- **LangChain Text Splitters** - Intelligent document chunking

## 🔧 Installation & Setup

### Prerequisites

- Python 3.13+
- API Keys for:
  - Pinecone (vector database)
  - Groq (LLM inference)
  - Tavily (web search)

### 1. Clone the Repository

```bash
git clone https://github.com/gauravramachandra/RAG_LangChain.git
cd RAG_LangChain
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Setup

Edit the `.env` file in the root directory:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=rag-index
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
DOC_SOURCE_DIR=data
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### 4. Run the Application

#### Start Backend Server

```bash
cd backend
uvicorn main:app --reload
```

#### Start Frontend (in another terminal)

```bash
cd frontend
streamlit run app.py
```

The application will be available at:

- **Frontend**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 🎯 How It Works

### Agent Workflow

1. **Router Node**: Analyzes user query and decides routing strategy

   - `rag`: Search internal knowledge base
   - `web`: Perform web search
   - `answer`: Direct response
   - `end`: Handle greetings/small talk

2. **RAG Lookup**: Retrieves relevant chunks from vector database

   - Uses semantic similarity search
   - Judges if retrieved content is sufficient

3. **Web Search**: Fetches real-time information from the internet

   - Triggered when RAG content is insufficient
   - Can be disabled per query

4. **Answer Generation**: Synthesizes final response using:
   - Retrieved documents
   - Web search results
   - LLM reasoning

### API Endpoints

#### POST `/upload-document/`

Upload and process PDF documents for the knowledge base.

#### POST `/chat/`

Main chat endpoint with intelligent routing.

```json
{
  "session_id": "user123",
  "query": "What is machine learning?",
  "enable_web_search": true
}
```

#### GET `/health`

Health check endpoint.

## 🔍 Key Features Explained

### Intelligent Routing

The agent automatically decides the best information source:

- **Internal docs** for specific, domain knowledge
- **Web search** for current events, recent information
- **Direct answering** for simple queries
- **Greeting handling** for conversational inputs

### Vector Database Integration

- Documents are chunked using `RecursiveCharacterTextSplitter`
- Embedded using HuggingFace sentence transformers
- Stored in Pinecone for fast similarity search
- Retrieval augments LLM responses with relevant context

### Session Management

- Persistent conversation history
- Thread-based context maintenance
- Memory-efficient checkpointing with LangGraph

## 🌟 Usage Examples

### Document Upload

```python
# Upload a PDF through the API
files = {'file': open('document.pdf', 'rb')}
response = requests.post('http://localhost:8000/upload-document/', files=files)
```

### Chat Query

```python
# Send a chat message
data = {
    "session_id": "user123",
    "query": "Explain quantum computing",
    "enable_web_search": True
}
response = requests.post('http://localhost:8000/chat/', json=data)
```
