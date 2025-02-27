import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.services.pdf_processing import load_reports
from rank_bm25 import BM25Okapi

# Load embedding model
embedding_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

# FAISS storage paths
FAISS_INDEX_DIR = "models"
FAISS_INDEX_PATH = os.path.join(FAISS_INDEX_DIR, "vector_store")

def process_documents():
    global vector_store, bm25_index
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
    
    # Compute embeddings
    embeddings_np = np.array(embedding_model.encode([chunk[0] for chunk in chunks], convert_to_tensor=False), dtype=np.float32)
    embeddings_np /= np.linalg.norm(embeddings_np, axis=1, keepdims=True)
    
    # Create and save FAISS index
    index = faiss.IndexFlatIP(embeddings_np.shape[1])
    index.add(embeddings_np)
    faiss.write_index(index, FAISS_INDEX_PATH)
    print("FAISS index created and saved successfully!")
    
    # Precompute BM25 index
    tokenized_chunks = [chunk[0].split(" ") for chunk in chunks]
    bm25_index = BM25Okapi(tokenized_chunks)
    print("BM25 index created successfully!")
    
    global vector_store
    vector_store = chunks

def load_faiss_index():
    """
    Loads the FAISS index from disk if available.

    Returns:
        faiss.Index or None: The loaded FAISS index if available, otherwise None.
    
    Warnings:
        - If the FAISS index file does not exist, a warning is printed.
        - Errors during FAISS index loading are caught and printed.
    """
    if not os.path.exists(FAISS_INDEX_PATH):
        print("Warning: FAISS index not found.")
        return None
    try:
        return faiss.read_index(FAISS_INDEX_PATH)
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None

def retrieve_relevant_chunks(query, faiss_index, bm25_index, k=5):
    if faiss_index is None or bm25_index is None:
        return []

    query_embedding_np = np.array([embedding_model.encode(query, convert_to_tensor=False)], dtype=np.float32)
    distances, indices = faiss_index.search(query_embedding_np, k)
    
    # Use precomputed BM25 index
    faiss_scores = [bm25_index.get_scores(query.split())[i] for i in indices[0]]
    ranked_indices = [x for _, x in sorted(zip(faiss_scores, indices[0]), reverse=True)][:k]
    retrieved_texts = [{"text": vector_store[i][0], "source": vector_store[i][1], "page": vector_store[i][2]} for i in ranked_indices]
    
    return retrieved_texts

def expand_context(retrieved_indices, chunks, window=1):
    """
    Expands retrieved text chunks by including neighboring chunks for additional context.

    Args:
        retrieved_indices (list): List of indices corresponding to retrieved text chunks.
        chunks (list): The full list of text chunks.
        window (int, optional): The number of chunks to include before and after each retrieved index. Defaults to 1.

    Returns:
        list: A list of expanded text strings combining neighboring chunks.
    
    Example:
        If `window=1` and index 5 is retrieved, chunks 4, 5, and 6 will be included in the result.
    """
    expanded_texts = []
    for idx in retrieved_indices:
        start = max(0, idx - window)
        end = min(len(chunks), idx + window + 1)
        expanded_texts.append("\n".join(chunks[start:end]))
    return expanded_texts
