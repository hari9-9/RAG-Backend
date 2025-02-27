from fastapi import APIRouter, HTTPException
from app.services.rag_pipeline import retrieve_relevant_chunks
import requests
import os
from dotenv import load_dotenv
import time
from fastapi import Request

# Load environment variables
load_dotenv()
HUGGING_FACE_KEY = os.getenv("HUGGING_FACE_KEY")

if not HUGGING_FACE_KEY:
    raise ValueError("Missing Hugging Face API key. Please set HUGGING_FACE_KEY in your .env file.")

# Create an API router for query-related endpoints
router = APIRouter(prefix="/query", tags=["Query"])

# Hugging Face API URL for inference (Change model if needed)
HF_API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"

# Headers for authentication with Hugging Face API
HEADERS = {
    "Authorization": f"Bearer {HUGGING_FACE_KEY}"
}

def call_huggingface_api(payload, max_retries=3, retry_delay=5):
    """
    Calls the Hugging Face Inference API with retry logic in case of failures.

    Args:
        payload (dict): The input payload containing the context and query for inference.
        max_retries (int, optional): Maximum number of retries if the API request fails. Defaults to 3.
        retry_delay (int, optional): Time in seconds to wait before retrying. Defaults to 5.

    Returns:
        dict: The JSON response from the Hugging Face API containing the generated text.

    Raises:
        HTTPException: If the API fails after multiple retries or encounters an error.
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
async def query_insights(request: Request, query: str):
    """
    Handles user queries and returns relevant insights extracted from reports.

    - Retrieves the most relevant text chunks from stored reports using `retrieve_relevant_chunks()`.
    - Formats the retrieved text for input into the Hugging Face model.
    - Calls the Hugging Face API to generate a concise answer.
    - Cleans and extracts the relevant response before returning it.

    Args:
        query (str): The user-provided query to search relevant information.

    Returns:
        dict: A dictionary containing:
            - "query" (str): The original user query.
            - "response" (str): The AI-generated response based on retrieved reports.
            - "sources" (list): List of relevant text chunks with metadata.

    Raises:
        HTTPException:
            - 400 if the query is empty.
            - 404 if no relevant information is found.
            - 500 if the Hugging Face API encounters an error.
    """
    print("Hit Get Method : Query")
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query parameter cannot be empty.")

    faiss_index = request.app.state.faiss_index
    bm25_index = request.app.state.bm25_index
    if faiss_index is None or bm25_index is None:
        raise HTTPException(status_code=500, detail="FAISS or BM25 index not loaded.")
    retrieved_texts = retrieve_relevant_chunks(query, faiss_index, bm25_index)

    if not retrieved_texts:
        raise HTTPException(status_code=404, detail="No relevant information found in the reports.")

    # Extract only the text from retrieved sources
    formatted_text = "\n\n".join([source["text"] for source in retrieved_texts])

    # Prepare payload for Hugging Face API
    payload = {
        "inputs": f"Based on the following information, answer the question concisely.\n\nContext:\n{formatted_text}\n\nQuestion: {query}\n\nAnswer:",
        "parameters": {"max_length": 200, "temperature": 0.1}  # Low temperature to minimize hallucinations
    }

    # Call Hugging Face API with retries
    result = call_huggingface_api(payload)
    raw_response = result[0]["generated_text"] if isinstance(result, list) and "generated_text" in result[0] else "No valid response."
    
    # Process the response to clean unnecessary formatting
    cleaned_response = clean_generated_response(raw_response)
    answer = extract_clean_answer(cleaned_response)
    
    return {
        "query": query,
        "response": answer,
        "sources": retrieved_texts  # Includes text, source filename, and page number
    }


def clean_generated_response(text):
    """
    Cleans the generated response by removing unwanted formatting.

    Args:
        text (str): The raw text output from the Hugging Face API.

    Returns:
        str: The cleaned text with unnecessary whitespace and newlines removed.
    """
    return text.replace("\n", " ").strip()


def extract_clean_answer(raw_response):
    """
    Extracts and returns only the answer from the model-generated response.

    - If the model explicitly includes "Answer:" in its response, it extracts the text after it.
    - Otherwise, it returns the response as-is.

    Args:
        raw_response (str): The raw generated response from the Hugging Face model.

    Returns:
        str: The extracted answer.
    """
    if "Answer:" in raw_response:
        return raw_response.split("Answer:")[-1].strip()
    return raw_response.strip()  # If no "Answer:", return as-is
