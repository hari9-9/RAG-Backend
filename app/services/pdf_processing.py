import pdfplumber
import os

# Ensure the correct path to the 'data/' folder
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Get the 'app/' directory
DATA_DIR = os.path.join(BASE_DIR, "data")  # Points to 'app/data/'

def extract_text_from_pdf(pdf_filename):
    """
    Extracts text from a given PDF file along with page numbers.

    - Opens the PDF using `pdfplumber`.
    - Iterates through each page, extracting the text if available.
    - Returns a list of tuples containing the extracted text and corresponding page numbers.

    Args:
        pdf_filename (str): The name of the PDF file to be processed.

    Returns:
        list of tuples or None:
            - A list where each tuple contains:
                - text (str): Extracted text from a page.
                - page_num (int): Page number where the text was found.
            - Returns `None` if the file is missing or text extraction fails.

    Warnings:
        - Prints a warning if the specified PDF file is not found.
        - Catches and prints any errors encountered while reading the PDF.

    Example:
        ```python
        extract_text_from_pdf("example.pdf")
        # Output: [("Page 1 text...", 1), ("Page 2 text...", 2), ...]
        ```
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
    Loads and extracts text from predefined PDF reports for processing in a Retrieval-Augmented Generation (RAG) pipeline.

    - Calls `extract_text_from_pdf()` for each predefined report.
    - Filters out reports that fail to load or return `None`.
    - Returns a dictionary containing the extracted text organized by filenames.

    Returns:
        dict:
            - Keys: PDF filenames.
            - Values: Lists of tuples containing extracted text and corresponding page numbers.

    Warnings:
        - If no reports are successfully loaded, a warning is printed.

    Example:
        ```python
        load_reports()
        # Output:
        # {
        #   "2023-conocophillips-aim-presentation.pdf": [("Page 1 text...", 1), ("Page 2 text...", 2)],
        #   "2024-conocophillips-proxy-statement.pdf": [("Page 1 text...", 1), ("Page 3 text...", 3)]
        # }
        ```
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
