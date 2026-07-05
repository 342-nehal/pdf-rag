## PROCESS OVERVIEW
# 1 upload pdf, select llm model and embedding model(streamlit)
# 2 extract text from pdf - SimplePDFProcessor class
# 3 make chunks of text, with ids and metadata - SimplePDFProcessor class
# 4 create collection with a specific embedding function
# 5 add documents to the collection with metadata, each documents having data of each chunk
# 6 ask a query
# 7 retrieve similar docs from collection
# 8 augment the query with resources
# 9 pass into a specific llm model
# 10 get the response



import PyPDF2
import uuid
CHUNK_SIZE=2000
CHUNK_OVERLAP=50
class SimplePDFProcessor:
    """Handle PDF processing and chunking"""

    def __init__(self, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def read_pdf(self, pdf_file):
        """Read PDF and extract text"""
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    def create_chunks(self, text, pdf_file):
        """Split text into chunks"""
        chunks = []
        start = 0

        while start < len(text):
            # Find end of chunk
            end = start + self.chunk_size

            # If not at the start, include overlap
            if start > 0:
                start = start - self.chunk_overlap

            # Get chunk
            chunk = text[start:end]

            # Try to break at sentence end
            if end < len(text):
                last_period = chunk.rfind(".")
                if last_period != -1:
                    chunk = chunk[: last_period + 1]
                    end = start + last_period + 1

            chunks.append(
                {
                    "id": str(uuid.uuid4()),  # cdefield24482kuy
                    "text": chunk,
                    "metadata": {"source":pdf_file.name},
                }
            )

            start = end

        return chunks
    
# pdf_processor=SimplePDFProcessor()

# text_extracted=pdf_processor.read_pdf(pdf_file="/Users/nehal/IdeaProjects/Chroma-db/DeepSeek_R1.pdf")
# print(text_extracted)
# chunks=pdf_processor.create_chunks(text_extracted, pdf_file="/Users/nehal/IdeaProjects/Chroma-db/DeepSeek_R1.pdf")
# print("printing chunks", chunks[0])
from chromadb.utils import embedding_functions
import streamlit as st
class SimpleModelSelector:
    """Simple class to handle model selection"""

    def __init__(self):
        # Available LLM models
        self.llm_models = {
            "ollama-llama": "llama3.2",
            "ollama-qwen":"qwen3:0.6b"     
        }
        

        # Available embedding models with their dimensions
        self.embedding_models = {
            "sentence-transformers-all-mpnet-base-v2": {
                "name": "all-mpnet-base-v2",
                "dimensions": 768,
                "model_name": None
            },
            "chroma-default-or-sentence-transformers-default": {
                "name": "all-MiniLM-L6-v2", 
                "dimensions": 384, 
                "model_name": None
            },
            "onnx-models": {
                "name": "all-MiniLM-L6-v2-onnx",
                "dimensions": 384,
                "model_name": None,
            },
        }

    def select_models(self):
        """Let user select models through Streamlit UI"""
        st.sidebar.title("📚 Model Selection")

        # Select LLM
        llm = st.sidebar.radio(
            "Choose LLM Model:",
            options=list(self.llm_models.keys()),
            format_func=lambda x: self.llm_models[x],
        )

        # Select Embeddings
        embedding = st.sidebar.radio(
            "Choose Embedding Model:",
            options=list(self.embedding_models.keys()),
            format_func=lambda x: self.embedding_models[x]["name"],
        )

        return llm, embedding
    
import chromadb
import ollama
class SimpleRagSystem:
    """Simple rag impl"""
    def __init__(self, embedding_model="chroma-default-or-sentence-transformers-default", llm_model="ollama-llama"):
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        
        self.db_client=chromadb.PersistentClient("./db/pdf_chroma_db")
        #set up embedding function
        if self.embedding_model=="sentence-transformers-all-mpnet-base-v2":  
            self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction("all-mpnet-base-v2")
        elif self.embedding_model=="chroma-default-or-sentence-transformers-default":
            self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction()
        else:
            self.embedder = embedding_functions.ONNXMiniLM_L6_V2()
        #set up llm model
        if self.llm_model=="ollama-llama":
            self.llm="llama3.2"
        elif self.llm_model=="ollama-qwen":
            self.llm="qwen3:0.6b"
        self.collection = None
    def generate_collection(self):
        """generate collection"""
        print("hello in generate collection")
        collection_name = f"documents_{self.embedding_model}"

        try:
            # Try to get existing collection first
            try:
                collection = self.db_client.get_collection(
                    name=collection_name, embedding_function=self.embedder
                )
                st.info(
                    f"Using existing collection for {self.embedding_model} embeddings"
                )
            except:
                # If collection doesn't exist, create new one
                collection = self.db_client.create_collection(
                    name=collection_name,
                    embedding_function=self.embedder,
                    metadata={"model": self.embedding_model},
                )
                st.success(
                    f"Created new collection for {self.embedding_model} embeddings"
                )

            return collection

        except Exception as e:
            st.error(f"Error setting up collection: {str(e)}")
            raise e
    
    def generate_documents(self, chunks):
        """add chunks to collection"""
        try:
            print("hello in generate docs")
            # Ensure collection exists
            if not self.collection:
                self.collection = self.generate_collection()

            # Add documents
            self.collection.add(
                ids=[chunk["id"] for chunk in chunks],
                documents=[chunk["text"] for chunk in chunks],
                metadatas=[chunk["metadata"] for chunk in chunks],
            )
            return True
        except Exception as e:
            st.error(f"Error adding documents: {str(e)}")
            return False
    def retrieve_relevant_docs(self, query, n_results=3):
        """query documents and return relevant chunks from collection"""
        return self.collection.query(query_texts=[query], n_results=n_results)
    def generate_response(self, query, context):
        """generate response using llm"""
        prompt=f"""Use the context to answer the query.
        context: {context}
        query: {query}
        answer:
        """
        response=ollama.chat(
            model=self.llm,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        return response["message"]["content"] 
    def get_embedding_info(self):
        """get info about embedding model"""
        model_selected=SimpleModelSelector()
        model_info=model_selected.embedding_models[self.embedding_model]
        return {
            "name": model_info["name"],
            "dimensions": model_info["dimensions"],
            "model": self.embedding_model
            }
def main():
    st.title("🤖 Simple RAG System")

    # Initialize session state
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    if "current_embedding_model" not in st.session_state:
        st.session_state.current_embedding_model = None
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None

    # Initialize model selector
    model_selector = SimpleModelSelector()
    llm_model, embedding_model = model_selector.select_models()

    # Check if embedding model changed
    if embedding_model != st.session_state.current_embedding_model:
        st.session_state.processed_files.clear()  # Clear processed files
        st.session_state.current_embedding_model = embedding_model
        st.session_state.rag_system = None  # Reset RAG system
        st.warning("Embedding model changed. Please re-upload your documents.")

    # Initialize RAG system
    try:
        if st.session_state.rag_system is None:
            st.session_state.rag_system = SimpleRagSystem(embedding_model, llm_model)

        # Display current embedding model info
        embedding_info = st.session_state.rag_system.get_embedding_info()
        st.sidebar.info(
            f"Current Embedding Model:\n"
            f"- Name: {embedding_info['name']}\n"
            f"- Dimensions: {embedding_info['dimensions']}"
        )
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return

    # File upload
    pdf_file = st.file_uploader("Upload PDF", type="pdf")

    if pdf_file and pdf_file.name not in st.session_state.processed_files:
        # Process PDF
        processor = SimplePDFProcessor()
        with st.spinner("Processing PDF..."):
            try:
                # Extract text
                text = processor.read_pdf(pdf_file)
                # Create chunks
                chunks = processor.create_chunks(text, pdf_file)
                # Add to database
                if st.session_state.rag_system.generate_documents(chunks):
                    st.session_state.processed_files.add(pdf_file.name)
                    st.success(f"Successfully processed {pdf_file.name}")
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")

    # Query interface
    if st.session_state.processed_files:
        st.markdown("---")
        st.subheader("🔍 Query Your Documents")
        query = st.text_input("Ask a question:")

        if query:
            with st.spinner("Generating response..."):
                # Get relevant chunks
                results = st.session_state.rag_system.retrieve_relevant_docs(query)
                print(results)
                if results and results["documents"]:
                    # Generate response
                    response = st.session_state.rag_system.generate_response(
                        query, results["documents"][0]
                    )

                    if response:
                        # Display results
                        st.markdown("### 📝 Answer:")
                        st.write(response)

                        with st.expander("View Source Passages"):
                            for idx, doc in enumerate(results["documents"][0], 1):
                                st.markdown(f"**Passage {idx}:**")
                                st.info(doc)
    else:
        st.info("👆 Please upload a PDF document to get started!")


if __name__ == "__main__":
    main()