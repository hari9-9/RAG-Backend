import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.services.pdf_processing import load_reports

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# FAISS storage location
FAISS_INDEX_DIR = "models"
FAISS_INDEX_PATH = "models/vector_store"

def process_documents():
    """
    Load, split, and store reports in FAISS for retrieval.
    """
    global vector_store

    # Load extracted text from reports
    reports = load_reports()
    full_text = reports["report_1"] + "\n" + reports["report_2"]

    # Split text into smaller chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(full_text)

    # Generate embeddings using SentenceTransformers
    embeddings = embedding_model.encode(chunks, convert_to_tensor=False)  # Ensure it returns a NumPy array

    # Convert embeddings list to NumPy array
    embeddings_np = np.array(embeddings, dtype=np.float32)

    # Initialize FAISS index and add embeddings
    index = faiss.IndexFlatL2(embeddings_np.shape[1])  # L2 distance metric
    index.add(embeddings_np)

    # Save FAISS index
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)

    print("✅ FAISS index created and saved successfully!")

def load_faiss_index():
    """
    Load FAISS index from disk if available.
    """
    if os.path.exists(FAISS_INDEX_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        return index
    return None

def retrieve_relevant_chunks(query, k=3):
    """
    Retrieve the most relevant document chunks for a given query using FAISS.
    """
    global vector_store

    # Load FAISS index if not already loaded
    index = load_faiss_index()
    if index is None:
        print("⚠️ Vector store is empty, initializing FAISS now...")
        process_documents()
        index = load_faiss_index()

    # Convert query into embedding
    query_embedding = embedding_model.encode(query, convert_to_tensor=False)
    query_embedding_np = np.array([query_embedding], dtype=np.float32)

    # Search for the most relevant chunks
    distances, indices = index.search(query_embedding_np, k)

    # Load extracted text to return the actual text chunks
    reports = load_reports()
    full_text = reports["report_1"] + "\n" + reports["report_2"]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(full_text)

    # Retrieve relevant text chunks
    retrieved_texts = [chunks[i] for i in indices[0] if i < len(chunks)]

    return retrieved_texts
