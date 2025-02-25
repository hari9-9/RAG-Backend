# RAG Market Research Tool

## Overview
This project is a **Retrieval-Augmented Generation (RAG) system** that allows users to analyze and compare two market research reports using AI-powered interactions. The system extracts relevant insights from the reports and generates responses to user queries while providing source citations.
This project integrates with the [RAG FrontEnd](https://github.com/hari9-9/RAG-Frontend)

## Features
- **Retrieval-Augmented Generation (RAG) Pipeline** for contextual AI-powered responses.
- **FastAPI Backend** for handling API requests efficiently.
- **Hugging Face Language Model Integration** for natural language understanding and response generation.
- **Vector Search with FAISS** for fast and scalable text retrieval.
- **BM25 Re-Ranking** to improve the relevance of retrieved text chunks.
- **PDF Processing** using `pdfplumber` to extract and index report contents.
- **Frontend Integration** (CORS enabled for communication with a frontend at `http://localhost:5173`).
- **Testing Suite** using `pytest` for API endpoints, PDF processing, and RAG pipeline validation.
- **Deployable on Render** using a `render.yaml` configuration file.

## Tech Stack
### **Backend:**
- **Language:** Python
- **Framework:** FastAPI
- **AI Models:** Hugging Face (`HuggingFaceH4/zephyr-7b-beta` for inference)
- **Vector Search:** FAISS
- **Text Embeddings:** `sentence-transformers`
- **BM25 Ranking:** `rank-bm25`
- **PDF Processing:** `pdfplumber`
- **Database:** N/A (Uses FAISS for efficient vector storage)

### **Frontend:**
- The backend is designed to work with any frontend framework (React/Vue/Next.js) via REST API.
- CORS configured for communication with a frontend running at `http://localhost:5173`.

### **Deployment & Hosting:**
- Deployed using **Render** (See `render.yaml` for configuration).

## Installation & Setup
### **1. Clone the Repository**
```sh
git clone https://github.com/your-username/rag-market-research-tool.git
cd rag-market-research-tool
```

### **2. Set Up the Backend**
#### **Install Dependencies:**
```sh
pip install -r requirements.txt
```

#### **Set Environment Variables:**
Create a `.env` file in the `app/` directory and add your Hugging Face API key:
```
HUGGING_FACE_KEY=your_api_key_here
```

#### **Run the Backend Server:**
```sh
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

### **3. Running Tests**
Run the test suite to validate API, PDF processing, and retrieval logic:
```sh
pytest tests/
```

## API Endpoints
### **1. Query API**
#### **GET `/query`**
Handles user queries and retrieves relevant insights from the reports.

**Request:**
```sh
GET http://localhost:8000/query?query=What is the revenue of the company?
```

**Response:**
```json
{
  "query": "What is the revenue of the company?",
  "response": "The company's revenue for 2023 was $10 billion.",
  "sources": [
    {"text": "Revenue for 2023 was reported as $10 billion.", "source": "2023-conocophillips-aim-presentation.pdf", "page": 3}
  ]
}
```

## Architecture Overview
### **1. RAG Pipeline Workflow**
1. Extracts and splits report text using `pdfplumber` and `RecursiveCharacterTextSplitter`.
2. Embeds text using `sentence-transformers` and stores it in **FAISS**.
3. On a query, retrieves top matches using **FAISS** and re-ranks results using **BM25**.
4. Formats the retrieved text and sends it to the **Hugging Face API** for AI-powered insights.
5. Returns a response along with the source data (document and page number).

### **2. FastAPI Backend Structure**
```
app/
│── main.py           # FastAPI app setup with CORS
│── routes/
│   ├── query.py      # Query endpoint and AI integration
│── services/
│   ├── pdf_processing.py  # Extracts text from PDFs
│   ├── rag_pipeline.py    # FAISS retrieval & BM25 ranking
│── data/             # Directory to store PDFs
│── models/           # FAISS index storage
```


## Future Enhancements
- **Advanced NLP Features:** Sentiment analysis, topic modeling.
- **Interactive UI:** Graphical insights with charting libraries.
- **Additional AI Models:** Experiment with different LLMs for better accuracy.
- **Improved Performance:** Optimize embedding storage and retrieval efficiency.


