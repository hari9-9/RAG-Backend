from fastapi import FastAPI
from app.routes.query import router as query_router
from fastapi.middleware.cors import CORSMiddleware

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
