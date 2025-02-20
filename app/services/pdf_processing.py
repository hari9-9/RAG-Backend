import pdfplumber
import os

# Ensure the correct path to 'data/' folder
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Get the 'app/' directory
DATA_DIR = os.path.join(BASE_DIR, "data")  # Points to 'app/data/'

def extract_text_from_pdf(pdf_filename):
    """
    Extracts text from a given PDF file.
    Handles missing files gracefully.
    """
    pdf_path = os.path.join(DATA_DIR, pdf_filename)

    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"Warning: File '{pdf_filename}' not found in {DATA_DIR}. Skipping.")
        return None

    text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = [page.extract_text() for page in pdf.pages if page.extract_text()]
    except Exception as e:
        print(f"Error reading {pdf_filename}: {e}")
        return None

    return "\n".join(text) if text else None  # Return None if extraction failed

def load_reports():
    """
    Extracts text from both reports and stores them for RAG processing.
    Handles missing reports safely.
    """
    reports = {
        "report_1": extract_text_from_pdf("2023-conocophillips-aim-presentation.pdf"),
        "report_2": extract_text_from_pdf("2024-conocophillips-proxy-statement.pdf")
    }

    # Remove reports that failed to load
    reports = {key: value for key, value in reports.items() if value is not None}

    if not reports:
        print("Warning: No reports loaded. Please check if PDFs exist in the 'data/' folder.")
    
    return reports
