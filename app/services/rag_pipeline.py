import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.services.pdf_processing import load_reports
from rank_bm25 import BM25Okapi

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

    # Improved chunking strategy
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(full_text)

    if not chunks:
        print("Warning: No text extracted for indexing. FAISS indexing skipped.")
        return None

    # Generate embeddings
    embeddings_np = np.array(embedding_model.encode(chunks, convert_to_tensor=False), dtype=np.float32)

    # Normalize embeddings for better FAISS performance
    embeddings_np /= np.linalg.norm(embeddings_np, axis=1, keepdims=True)

    # Initialize FAISS index (use HNSW for better recall)
    index = faiss.IndexFlatIP(embeddings_np.shape[1])
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
        print("Warning: FAISS index not found.")
        return None
    try:
        return faiss.read_index(FAISS_INDEX_PATH)
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None

def retrieve_relevant_chunks(query, k=5):
    """
    Retrieve the most relevant document chunks for a given query using FAISS + BM25 Hybrid Search.
    """
    # Load FAISS index
    index = load_faiss_index()
    if index is None:
        print("Vector store is empty, initializing FAISS now...")
        process_documents()
        index = load_faiss_index()
        if index is None:
            return []

    # Convert query into embedding
    query_embedding_np = np.array([embedding_model.encode(query, convert_to_tensor=False)], dtype=np.float32)

    # FAISS dense retrieval
    distances, indices = index.search(query_embedding_np, k*2)

    # Load extracted text
    reports = load_reports()
    if not reports:
        return []

    full_text = "\n".join(reports.values())
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(full_text)

    # BM25 sparse retrieval
    tokenized_chunks = [chunk.split(" ") for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    faiss_chunks = [chunks[i] for i in indices[0]]
    faiss_scores = [bm25.get_scores(query.split())[i] for i in indices[0]]


    # Merge FAISS and BM25 results
     # Sort FAISS results by BM25 relevance
    ranked_indices = [x for _, x in sorted(zip(faiss_scores, indices[0]), reverse=True)][:k]
    
    # Retrieve relevant text chunks
    retrieved_texts = [chunks[i] for i in ranked_indices]


    return retrieved_texts


def expand_context(retrieved_indices, chunks, window=1):
    expanded_texts = []
    for idx in retrieved_indices:
        start = max(0, idx - window)
        end = min(len(chunks), idx + window + 1)
        expanded_texts.append("\n".join(chunks[start:end]))
    return expanded_texts