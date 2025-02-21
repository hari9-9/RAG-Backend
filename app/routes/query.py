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
HF_API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"




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
    formatted_text = "\n\n".join(retrieved_texts)
    payload = {
        "inputs": f"Based on the following information, answer the question concisely.\n\nContext:\n{formatted_text}\n\nQuestion: {query}\n\nAnswer:",
        "parameters": {"max_length": 200, "temperature": 0.1}  # Prevent hallucination
    }

    print("Payload Sent to LLM:\n", payload)


    # Call Hugging Face API with retries
    result = call_huggingface_api(payload)
    raw_response = result[0]["generated_text"] if isinstance(result, list) and "generated_text" in result[0] else "No valid response."
    cleaned_response = clean_generated_response(raw_response)
    answer = extract_clean_answer(cleaned_response)
    return {
        "query": query,
        "response": answer,
        "sources": retrieved_texts
    }



def clean_generated_response(text):
    # Remove unwanted words, unnecessary formatting
    text = text.replace("\n", " ").strip()
    return text

def extract_clean_answer(raw_response):
    """
    Extracts only the answer from the LLM-generated response.
    Removes unnecessary context repetition.
    """
    if "Answer:" in raw_response:
        return raw_response.split("Answer:")[-1].strip()
    return raw_response.strip()  # If no "Answer:", return as-is