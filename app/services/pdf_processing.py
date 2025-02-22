import pdfplumber
import os

# Ensure the correct path to 'data/' folder
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Get the 'app/' directory
DATA_DIR = os.path.join(BASE_DIR, "data")  # Points to 'app/data/'

def extract_text_from_pdf(pdf_filename):
    """
    Extracts text from a given PDF file along with page numbers.
    """
    pdf_path = os.path.join(DATA_DIR, pdf_filename)
    
    if not os.path.exists(pdf_path):
        print(f"Warning: File '{pdf_filename}' not found in {DATA_DIR}. Skipping.")
        return None
    
    extracted_text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    extracted_text.append((text, page_num))  # Store text with page number
    except Exception as e:
        print(f"Error reading {pdf_filename}: {e}")
        return None
    
    return extracted_text if extracted_text else None

def load_reports():
    """
    Extracts text from both reports and stores them for RAG processing.
    Handles missing reports safely.
    """
    reports = {
        "2023-conocophillips-aim-presentation.pdf": extract_text_from_pdf("2023-conocophillips-aim-presentation.pdf"),
        "2024-conocophillips-proxy-statement.pdf": extract_text_from_pdf("2024-conocophillips-proxy-statement.pdf")
    }

    # Remove reports that failed to load
    reports = {key: value for key, value in reports.items() if value is not None}

    if not reports:
        print("Warning: No reports loaded. Please check if PDFs exist in the 'data/' folder.")
    
    return reports
