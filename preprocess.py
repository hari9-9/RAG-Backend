# preprocess.py
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.services.pdf_processing import load_reports

# Load embedding model
embedding_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

# FAISS storage paths
FAISS_INDEX_DIR = "models"
FAISS_INDEX_PATH = os.path.join(FAISS_INDEX_DIR, "vector_store")
EMBEDDINGS_PATH = os.path.join(FAISS_INDEX_DIR, "embeddings.npy")

def preprocess_and_save():
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    
    # Load reports and split text
    reports = load_reports()
    if not reports:
        print("Warning: No reports available. Preprocessing skipped.")
        return

    chunks_with_metadata = []
    for filename, pages in reports.items():
        for text, page_num in pages:
            chunks_with_metadata.append((text, filename, page_num))
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = [(sub_chunk, filename, page_num) for text, filename, page_num in chunks_with_metadata for sub_chunk in splitter.split_text(text)]
    
    if not chunks:
        print("Warning: No text extracted for preprocessing. Skipping.")
        return
    
    # Compute embeddings
    embeddings_np = np.array(embedding_model.encode([chunk[0] for chunk in chunks], convert_to_tensor=False), dtype=np.float32)
    embeddings_np /= np.linalg.norm(embeddings_np, axis=1, keepdims=True)
    
    # Save embeddings
    np.save(EMBEDDINGS_PATH, embeddings_np)
    
    # Create and save FAISS index
    index = faiss.IndexFlatIP(embeddings_np.shape[1])
    index.add(embeddings_np)
    faiss.write_index(index, FAISS_INDEX_PATH)
    print("Preprocessing complete! Embeddings and FAISS index saved.")

if __name__ == "__main__":
    preprocess_and_save()