import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.services.pdf_processing import load_reports

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# FAISS storage paths
FAISS_INDEX_DIR = "models"
FAISS_INDEX_PATH = os.path.join(FAISS_INDEX_DIR, "vector_store")

def process_documents():
    """
    Load, split, and store reports in FAISS for retrieval.
    Handles missing reports gracefully.
    """
    global vector_store

    # Ensure FAISS directory exists
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

    # Load extracted text from reports
    reports = load_reports()
    if not reports:
        print("Warning: No reports available. FAISS indexing skipped.")
        return None

    full_text = "\n".join(reports.values())

    # Split text into smaller chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(full_text)

    if not chunks:
        print("Warning: No text extracted for indexing. FAISS indexing skipped.")
        return None

    # Generate embeddings
    embeddings_np = np.array(embedding_model.encode(chunks, convert_to_tensor=False), dtype=np.float32)

    # Initialize FAISS index and store embeddings
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)

    # Save FAISS index
    try:
        faiss.write_index(index, FAISS_INDEX_PATH)
        print("FAISS index created and saved successfully!")
    except Exception as e:
        print(f"Error saving FAISS index: {e}")

def load_faiss_index():
    """
    Load FAISS index from disk if available.
    """
    if not os.path.exists(FAISS_INDEX_PATH):
        print("⚠️ Warning: FAISS index not found.")
        return None
    try:
        return faiss.read_index(FAISS_INDEX_PATH)
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None

def retrieve_relevant_chunks(query, k=3):
    """
    Retrieve the most relevant document chunks for a given query using FAISS.
    """
    # Load FAISS index
    index = load_faiss_index()
    if index is None:
        print("⚠️ Vector store is empty, initializing FAISS now...")
        process_documents()
        index = load_faiss_index()
        if index is None:
            return []

    # Convert query into embedding
    query_embedding_np = np.array([embedding_model.encode(query, convert_to_tensor=False)], dtype=np.float32)

    # Search for the most relevant chunks
    distances, indices = index.search(query_embedding_np, k)

    # Load extracted text to return the actual text chunks
    reports = load_reports()
    if not reports:
        return []

    full_text = "\n".join(reports.values())
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(full_text)

    # Retrieve relevant text chunks (ensures indices are valid)
    retrieved_texts = [chunks[i] for i in indices[0] if i < len(chunks)]

    return retrieved_texts
