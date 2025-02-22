# tests/test_pdf_processing.py
import os
import tempfile
import pytest
from app.services import pdf_processing

def test_extract_text_from_pdf_nonexistent():
    # When the PDF does not exist, the function should return None.
    result = pdf_processing.extract_text_from_pdf("nonexistent.pdf")
    assert result is None

# Dummy classes to simulate a PDF and its pages
class DummyPage:
    def __init__(self, text):
        self._text = text
    def extract_text(self):
        return self._text

class DummyPDF:
    def __init__(self, pages_text):
        self.pages = [DummyPage(text) for text in pages_text]
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

# Dummy pdfplumber.open function that returns a DummyPDF instance
def dummy_pdf_open(pdf_path):
    # Simulate a PDF with two pages
    return DummyPDF(["Page 1 dummy text", "Page 2 dummy text"])

def test_extract_text_from_pdf_existing(monkeypatch, tmp_path):
    # Create a temporary data directory
    dummy_data_dir = tmp_path / "data"
    dummy_data_dir.mkdir()
    # Override the DATA_DIR in pdf_processing to point to our temp folder
    monkeypatch.setattr(pdf_processing, "DATA_DIR", str(dummy_data_dir))
    # Create a dummy PDF file (its content is irrelevant because we monkeypatch pdfplumber.open)
    dummy_pdf_file = dummy_data_dir / "dummy.pdf"
    dummy_pdf_file.write_text("dummy content")
    
    # Override pdfplumber.open to use our dummy_pdf_open function
    monkeypatch.setattr(pdf_processing, "pdfplumber", type("dummy", (), {"open": dummy_pdf_open}))
    
    result = pdf_processing.extract_text_from_pdf("dummy.pdf")
    assert result is not None
    # Expect two pages with correct page numbers and text
    assert len(result) == 2
    assert result[0][0] == "Page 1 dummy text"
    assert result[0][1] == 1

def test_load_reports(monkeypatch, tmp_path):
    dummy_data_dir = tmp_path / "data"
    dummy_data_dir.mkdir()
    monkeypatch.setattr(pdf_processing, "DATA_DIR", str(dummy_data_dir))

    # Create dummy PDFs with expected names
    (dummy_data_dir / "2023-conocophillips-aim-presentation.pdf").write_text("dummy content")
    (dummy_data_dir / "2024-conocophillips-proxy-statement.pdf").write_text("dummy content")

    # Override pdfplumber.open to simulate successful PDF reading
    monkeypatch.setattr(pdf_processing, "pdfplumber", type("dummy", (), {"open": dummy_pdf_open}))

    # Call load_reports() and verify expected reports are loaded
    reports = pdf_processing.load_reports()
    assert "2023-conocophillips-aim-presentation.pdf" in reports
    assert "2024-conocophillips-proxy-statement.pdf" in reports

