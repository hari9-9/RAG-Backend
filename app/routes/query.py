from fastapi import APIRouter, HTTPException
from app.services.rag_pipeline import retrieve_relevant_chunks
from langchain.llms import OpenAI

from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

router = APIRouter(prefix="/query", tags=["Query"])

@router.get("/")
async def query_insights(query: str):
    """
    Accepts a user query and returns insights from the two reports.
    """
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is required.")

    retrieved_texts = retrieve_relevant_chunks(query)
    
    if not retrieved_texts:
        raise HTTPException(status_code=404, detail="No relevant information found.")
    llm = OpenAI(api_key=OPENAI_API_KEY)
    # llm = OpenAI(model="gpt-4")
    response = llm(f"Analyze the following report sections and answer: {query}\n\n{retrieved_texts}")

    return {"query": query, "response": response, "sources": retrieved_texts}
