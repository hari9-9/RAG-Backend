# tests/test_api.py
import pytest
from fastapi.testclient import TestClient

# Import the FastAPI app
from app.main import app

# Import the query module so we can override functions
from app.routes import query

# Dummy functions to override actual implementations during tests
def dummy_retrieve_relevant_chunks(query_text):
    return [{"text": "Dummy context text.", "source": "dummy.pdf", "page": 1}]

def dummy_call_huggingface_api(payload, max_retries=3, retry_delay=5):
    # Simulate a response from the Hugging Face API
    return [{"generated_text": "Answer: Dummy answer generated."}]

# Use pytest's monkeypatch fixture to override functions in the module
@pytest.fixture(autouse=True)
def override_query_functions(monkeypatch):
    monkeypatch.setattr(query, "retrieve_relevant_chunks", dummy_retrieve_relevant_chunks)
    monkeypatch.setattr(query, "call_huggingface_api", dummy_call_huggingface_api)

client = TestClient(app)

def test_empty_query():
    response = client.get("/query", params={"query": ""})
    assert response.status_code == 400
    data = response.json()
    assert data["detail"] == "Query parameter cannot be empty."

def test_valid_query():
    response = client.get("/query", params={"query": "What is dummy?"})
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "What is dummy?"
    # Check that the dummy API returns the expected answer
    assert "Dummy answer generated." in data["response"]
    # Ensure at least one source is returned
    assert isinstance(data["sources"], list)
    assert len(data["sources"]) > 0
