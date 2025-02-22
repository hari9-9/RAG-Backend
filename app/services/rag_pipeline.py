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
    Load, split, and store reports in FAISS for retrieval with metadata.
    """
    global vector_store

    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    
    reports = load_reports()
    if not reports:
        print("Warning: No reports available. FAISS indexing skipped.")
        return None

    chunks_with_metadata = []
    
    for filename, pages in reports.items():
        for text, page_num in pages:
            chunks_with_metadata.append((text, filename, page_num))
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = [(sub_chunk, filename, page_num) for text, filename, page_num in chunks_with_metadata for sub_chunk in splitter.split_text(text)]
    
    if not chunks:
        print("Warning: No text extracted for indexing. FAISS indexing skipped.")
        return None
    
    embeddings_np = np.array(embedding_model.encode([chunk[0] for chunk in chunks], convert_to_tensor=False), dtype=np.float32)
    embeddings_np /= np.linalg.norm(embeddings_np, axis=1, keepdims=True)
    
    index = faiss.IndexFlatIP(embeddings_np.shape[1])
    index.add(embeddings_np)
    
    try:
        faiss.write_index(index, FAISS_INDEX_PATH)
        print("FAISS index created and saved successfully!")
    except Exception as e:
        print(f"Error saving FAISS index: {e}")
    
    global vector_store
    vector_store = chunks

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

def retrieve_relevant_chunks(query, k=2):
    """
    Retrieve the most relevant document chunks for a given query using FAISS + BM25 Hybrid Search.
    """
    index = load_faiss_index()
    if index is None:
        print("Vector store is empty, initializing FAISS now...")
        process_documents()
        index = load_faiss_index()
        if index is None:
            return []

    query_embedding_np = np.array([embedding_model.encode(query, convert_to_tensor=False)], dtype=np.float32)
    distances, indices = index.search(query_embedding_np, k*2)

    reports = load_reports()
    if not reports:
        return []

    chunks_with_metadata = []
    for filename, pages in reports.items():
        for text, page_num in pages:
            chunks_with_metadata.append((text, filename, page_num))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = [(sub_chunk, filename, page_num) for text, filename, page_num in chunks_with_metadata for sub_chunk in splitter.split_text(text)]
    
    tokenized_chunks = [chunk[0].split(" ") for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    faiss_chunks = [chunks[i] for i in indices[0]]
    faiss_scores = [bm25.get_scores(query.split())[i] for i in indices[0]]
    
    ranked_indices = [x for _, x in sorted(zip(faiss_scores, indices[0]), reverse=True)][:k]
    retrieved_texts = [{"text": chunks[i][0], "source": chunks[i][1], "page": chunks[i][2]} for i in ranked_indices]
    
    return retrieved_texts



def expand_context(retrieved_indices, chunks, window=1):
    expanded_texts = []
    for idx in retrieved_indices:
        start = max(0, idx - window)
        end = min(len(chunks), idx + window + 1)
        expanded_texts.append("\n".join(chunks[start:end]))
    return expanded_texts