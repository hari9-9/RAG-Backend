# tests/test_rag_pipeline.py
import os
import numpy as np
import pytest
from app.services import rag_pipeline

# Dummy load_reports function that returns predictable data
def dummy_load_reports():
    return {
        "dummy.pdf": [("This is dummy text for testing.", 1)]
    }

# A dummy text splitter that splits text into predictable chunks.
class DummySplitter:
    def __init__(self, chunk_size, chunk_overlap):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    def split_text(self, text):
        # For simplicity, split text into chunks of three words
        words = text.split()
        chunks = []
        for i in range(0, len(words), 3):
            chunks.append(" ".join(words[i:i+3]))
        return chunks

# Dummy embedding function that returns a fixed vector of ones
def dummy_encode(texts, convert_to_tensor=False):
    # Assuming embedding dimension is 384 for the dummy model
    return [np.ones(384, dtype=np.float32) for _ in texts]

# Use a fixture to override dependencies and set up a temporary FAISS index directory
@pytest.fixture(autouse=True)
def setup_dummy(monkeypatch, tmp_path):
    # Override load_reports with our dummy version
    monkeypatch.setattr(rag_pipeline, "load_reports", dummy_load_reports)
    # Override the text splitter with our dummy splitter
    monkeypatch.setattr(rag_pipeline, "RecursiveCharacterTextSplitter", lambda chunk_size, chunk_overlap: DummySplitter(chunk_size, chunk_overlap))
    # Override the embedding model's encode method with our dummy implementation
    monkeypatch.setattr(rag_pipeline.embedding_model, "encode", dummy_encode)
    # Use a temporary directory for storing the FAISS index
    dummy_index_dir = tmp_path / "models"
    dummy_index_dir.mkdir()
    monkeypatch.setattr(rag_pipeline, "FAISS_INDEX_DIR", str(dummy_index_dir))
    dummy_index_path = dummy_index_dir / "vector_store"
    monkeypatch.setattr(rag_pipeline, "FAISS_INDEX_PATH", str(dummy_index_path))
    # Reset global vector_store if it exists
    if hasattr(rag_pipeline, "vector_store"):
        del rag_pipeline.vector_store

def test_process_documents_creates_index(monkeypatch, tmp_path):
    # Run process_documents and check that the FAISS index file is created
    rag_pipeline.process_documents()
    index_path = rag_pipeline.FAISS_INDEX_PATH
    assert os.path.exists(index_path)
    # Verify that the global vector_store is populated with chunks
    assert hasattr(rag_pipeline, "vector_store")
    assert isinstance(rag_pipeline.vector_store, list)
    assert len(rag_pipeline.vector_store) > 0

def test_expand_context():
    # Test that expand_context correctly expands the retrieved chunk based on the window parameter.
    dummy_chunks = ["Chunk 1", "Chunk 2", "Chunk 3", "Chunk 4", "Chunk 5"]
    retrieved_indices = [2]
    expanded = rag_pipeline.expand_context(retrieved_indices, dummy_chunks, window=1)
    # For index 2 with a window of 1, the expected context includes indices 1, 2, and 3.
    expected = "\n".join(["Chunk 2", "Chunk 3", "Chunk 4"])
    assert expanded[0] == expected
