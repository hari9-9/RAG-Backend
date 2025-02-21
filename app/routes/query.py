from fastapi import APIRouter, HTTPException
from app.services.rag_pipeline import retrieve_relevant_chunks
import requests
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()
HUGGING_FACE_KEY = os.getenv("HUGGING_FACE_KEY")

if not HUGGING_FACE_KEY:
    raise ValueError("Missing Hugging Face API key. Please set HUGGING_FACE_KEY in your .env file.")

router = APIRouter(prefix="/query", tags=["Query"])

# Hugging Face API URL (Change model if needed)
HF_API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"

# Headers for authentication
HEADERS = {
    "Authorization": f"Bearer {HUGGING_FACE_KEY}",
    "X-Use-Cache": "0"  # Forces fresh responses
}

def call_huggingface_api(payload, max_retries=3, retry_delay=5):
    """
    Calls Hugging Face API with retries in case of failure.
    """
    for attempt in range(max_retries):
        response = requests.post(HF_API_URL, headers=HEADERS, json=payload)

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503:  # Service unavailable
            print(f"Hugging Face API unavailable. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        elif response.status_code == 429:  # Rate limit exceeded
            print(f"Hugging Face rate limit reached. Waiting before retrying...")
            time.sleep(retry_delay)
        else:
            raise HTTPException(status_code=500, detail=f"Hugging Face API error: {response.json()}")

    raise HTTPException(status_code=500, detail="Hugging Face API failed after multiple retries.")


@router.get("/")
async def query_insights(query: str):
    """
    Accepts a user query and returns structured insights from the reports using Hugging Face's Inference API.
    """
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query parameter cannot be empty.")

    retrieved_texts = retrieve_relevant_chunks(query)

    if not retrieved_texts:
        raise HTTPException(status_code=404, detail="No relevant information found in the reports.")


    # Prepare payload for Hugging Face API
    payload = {
        "inputs": f"Analyze the following report sections and answer: {query}\n\n{retrieved_texts}",
        "parameters": {"max_length": 100, "do_sample": True}
    }

    # Call Hugging Face API with retries
    result = call_huggingface_api(payload)

    return {
        "query": query,
        "response": result[0]["generated_text"] if isinstance(result, list) and "generated_text" in result[0] else "No valid response generated.",
        "sources": retrieved_texts
    }

