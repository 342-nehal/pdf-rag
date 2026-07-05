# 1. Base image — a Linux machine with Python already installed
FROM python:3.11-slim
# Why slim? Smaller download, enough for most apps.
# Why 3.11? Stable; avoid 3.13 if some ML libs aren’t ready yet.

# 2. Working directory inside the container
# Why? All commands run from /app; your code lives here.
WORKDIR /app

# 3. Copy dependency list first (Docker caching trick)
COPY requirements.txt .
# Why copy requirements BEFORE code?
# Docker caches layers. If only your .py changes, it won't reinstall all pip packages.

# 4. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Why --no-cache-dir? Keeps the image smaller.

# 5. Copy your application code
COPY simple_rag_pdf.py .

# 6. Create folder for ChromaDB (matches your code: ./db/pdf_chroma_db)
RUN mkdir -p /app/db/pdf_chroma_db
# Why? Your code uses PersistentClient("./db/pdf_chroma_db").
# The folder must exist inside the container.

# 7. Tell Docker this app listens on port 8501 (Streamlit default)
EXPOSE 8501

# 8. Command to start the app when container starts
CMD ["streamlit", "run", "simple_rag_pdf.py", "--server.address", "0.0.0.0", "--server.port", "8501"]