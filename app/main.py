from fastapi import FastAPI
from app.routes.query import router as query_router

app = FastAPI(title="RAG Market Research Tool")

# Register routes
app.include_router(query_router)
