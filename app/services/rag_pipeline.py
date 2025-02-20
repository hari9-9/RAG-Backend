import faiss
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from app.services.pdf_processing import load_reports

from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print(OPENAI_API_KEY)
# Load extracted text from reports
reports = load_reports()

# Split text into smaller chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(reports["report_1"] + "\n" + reports["report_2"])

# Generate embeddings
embeddings_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
vector_store = FAISS.from_texts(chunks, embeddings_model)

# Save FAISS index
vector_store.save_local("models/vector_store")

def retrieve_relevant_chunks(query):
    """
    Retrieve the most relevant report chunks for a given query.
    """
    query_embedding = embeddings_model.embed_query(query)
    results = vector_store.similarity_search_by_vector(query_embedding, k=3)
    
    return [r.page_content for r in results]
