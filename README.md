# Simple RAG PDF

A lightweight **Retrieval-Augmented Generation (RAG)** app that lets you upload PDF documents, embed them into [ChromaDB](https://www.trychroma.com/), and ask questions answered by a local [Ollama](https://ollama.com/) LLM.

Built with Streamlit for a simple browser UI — no separate frontend required.

## Features

- Upload PDF files and automatically chunk + embed them
- Choose from multiple embedding models (Sentence Transformers, ONNX)
- Choose from local Ollama LLMs (Llama 3.2, Qwen 3)
- Query your documents with natural language
- View source passages used to generate each answer
- Persistent ChromaDB storage across sessions
- Docker Compose setup for easy deployment

## Tech Stack

| Layer | Technology |
|-------|------------|
| UI | [Streamlit](https://streamlit.io/) |
| Backend | Python 3.11 |
| Vector DB | [ChromaDB](https://www.trychroma.com/) (persistent client) |
| LLM | [Ollama](https://ollama.com/) |
| PDF parsing | PyPDF2 |
| Embeddings | sentence-transformers, ONNX MiniLM |

## How It Works

1. Upload a PDF and select an LLM + embedding model in the Streamlit sidebar
2. Extract text from the PDF (`SimplePDFProcessor`)
3. Split text into overlapping chunks with metadata (`SimplePDFProcessor`)
4. Create or reuse a ChromaDB collection for the selected embedding model
5. Add document chunks to the collection
6. Ask a question via the query interface
7. Retrieve the most similar chunks from ChromaDB
8. Augment the prompt with retrieved context
9. Send the prompt to the selected Ollama model
10. Display the answer and source passages

## Prerequisites

### Local development

- Python 3.11+
- [Ollama](https://ollama.com/download) installed and running
- Ollama models pulled locally:

```bash
ollama pull llama3.2
ollama pull qwen3:0.6b
```

### Docker deployment

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose

## Installation

### Option 1: Run locally

```bash
# Clone the repository
git clone https://github.com/342-nehal/pdf-rag.git
cd pdf-rag

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the app
streamlit run simple_rag_pdf.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Option 2: Run with Docker Compose

```bash
git clone https://github.com/342-nehal/pdf-rag.git
cd pdf-rag

docker compose up --build
```

After the containers start, pull the required Ollama models:

```bash
docker exec -it ollama ollama pull llama3.2
docker exec -it ollama ollama pull qwen3:0.6b
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Usage

1. Select an **LLM model** and **embedding model** from the sidebar
2. Upload a PDF using the file uploader
3. Wait for the document to be processed and indexed
4. Type a question in the query box and press Enter
5. Read the generated answer and expand **View Source Passages** to see retrieved context

> **Note:** Changing the embedding model clears previously uploaded documents — re-upload your PDFs after switching models.

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `CHUNK_SIZE` | 2000 | Characters per text chunk |
| `CHUNK_OVERLAP` | 50 | Overlap between consecutive chunks |
| ChromaDB path | `./db/pdf_chroma_db` | Local persistent storage directory |
| Streamlit port | 8501 | Web UI port |

## Project Structure

```
simple_rag_pdf/
├── simple_rag_pdf.py   # Main app (PDF processing, RAG pipeline, Streamlit UI)
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container image for the Streamlit app
├── docker-compose.yml  # Ollama + app orchestration
└── README.md
```

## Supported Models

**LLM (via Ollama)**

- `llama3.2`
- `qwen3:0.6b`

**Embeddings**

- `all-mpnet-base-v2` (768 dimensions)
- `all-MiniLM-L6-v2` (384 dimensions, ChromaDB default)
- `all-MiniLM-L6-v2-onnx` (384 dimensions, ONNX)