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
    Loads reports, splits them into smaller text chunks, and indexes them in FAISS for efficient retrieval.

    - Reports are loaded using `load_reports()`, returning a dictionary with filenames as keys and lists of 
      (text, page_number) tuples as values.
    - The text is split into smaller chunks using `RecursiveCharacterTextSplitter`.
    - Embeddings for the text chunks are computed using `SentenceTransformer` and stored in a FAISS index.
    - The FAISS index is then saved to disk for future retrieval.

    Warnings:
        - If no reports are found, indexing is skipped.
        - If no text is extracted, indexing is skipped.
        - Errors during FAISS index writing are caught and printed.

    Global Variables:
        vector_store (list): Stores processed chunks with metadata for retrieval.

    Returns:
        None
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

def retrieve_relevant_chunks(query, k=5):
    """
    Retrieves the most relevant document chunks for a given query using a hybrid FAISS + BM25 search.

    Steps:
    1. Loads the FAISS index from disk.
    2. Computes the query embedding using `SentenceTransformer` and searches for the top `2k` matches in FAISS.
    3. Loads the original reports and re-splits them using `RecursiveCharacterTextSplitter`.
    4. Uses BM25 to re-rank the FAISS results based on token similarity.
    5. Returns the top `k` ranked results.

    Args:
        query (str): The user query to search for relevant text chunks.
        k (int, optional): Number of relevant chunks to return. Defaults to 5.

    Returns:
        list: A list of dictionaries containing:
            - 'text' (str): The retrieved text chunk.
            - 'source' (str): The filename of the original document.
            - 'page' (int): The page number where the text chunk appears.

    Warnings:
        - If no FAISS index is found, it attempts to process documents.
        - If no reports are available, an empty list is returned.
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
