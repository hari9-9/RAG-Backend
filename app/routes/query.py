from fastapi import APIRouter, HTTPException
from app.services.rag_pipeline import retrieve_relevant_chunks
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HUGGING_FACE_KEY = os.getenv("HUGGING_FACE_KEY")

if not HUGGING_FACE_KEY:
    raise ValueError("Missing Hugging Face API key. Please set HUGGING_FACE_KEY in your .env file.")

router = APIRouter(prefix="/query", tags=["Query"])

# Hugging Face API URL (Change model if needed)
# HF_API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"

# HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"

HF_API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"


# Headers for authentication
HEADERS = {"Authorization": f"Bearer {HUGGING_FACE_KEY}"}

@router.get("/")
async def query_insights(query: str):
    """
    Accepts a user query and returns insights from the two reports using Hugging Face's Inference API.
    """
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is required.")

    retrieved_texts = retrieve_relevant_chunks(query)

    if not retrieved_texts:
        raise HTTPException(status_code=404, detail="No relevant information found.")

    # Prepare payload
    payload = {
        "inputs": f"Analyze the following report sections and answer: {query}\n\n{retrieved_texts}",
        "parameters": {"max_length": 500, "do_sample": True}
    }

    # Make request to Hugging Face API
    response = requests.post(HF_API_URL, headers=HEADERS, json=payload)

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Hugging Face API error: {response.json()}")

    result = response.json()

    return {"query": query, "response": result[0]["generated_text"], "sources": retrieved_texts}
