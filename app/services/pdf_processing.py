import pdfplumber
import os

# Ensure the correct path to 'data/' folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Get the 'app/' directory
DATA_DIR = os.path.join(BASE_DIR, "data")  # Points to 'app/data/'

def extract_text_from_pdf(pdf_filename):
    """
    Extracts text from a given PDF file.
    """
    text = ""
    pdf_path = os.path.join(DATA_DIR, pdf_filename)
    print(pdf_path)
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    
    return text

def load_reports():
    """
    Extracts text from both reports and stores them for RAG processing.
    """
    report_1_text = extract_text_from_pdf("2023-conocophillips-aim-presentation.pdf")
    report_2_text = extract_text_from_pdf("2024-conocophillips-proxy-statement.pdf")

    return {
        "report_1": report_1_text,
        "report_2": report_2_text
    }
