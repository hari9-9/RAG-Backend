from fastapi import FastAPI
from app.routes.query import router as query_router
from fastapi.middleware.cors import CORSMiddleware
from app.services.rag_pipeline import process_documents
from app.services.rag_pipeline import load_faiss_index
from app.services import rag_pipeline
app = FastAPI(title="RAG Market Research Tool")

# Register routes
app.include_router(query_router)

# Enable CORS for your frontend (React at localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Allow frontend to access API
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Preprocess documents and load indices
process_documents()
bm25_index = rag_pipeline.bm25_index
app.state.bm25_index = bm25_index


# Load FAISS index at startup
faiss_index = load_faiss_index()
if faiss_index is None:
    print("Warning: FAISS index not found. Please preprocess documents first.")
else:
    print("FAISS index loaded successfully!")

# Store the index in the app state
app.state.faiss_index = faiss_index