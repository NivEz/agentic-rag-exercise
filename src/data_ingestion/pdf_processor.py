"""PDF processing module for extracting text from insurance claim documents."""

from typing import Dict
from pathlib import Path
import pymupdf.layout
import pymupdf4llm


class PDFProcessor:
    """Processes PDF files and extracts text."""
    
    def __init__(self):
        """Initialize PDF processor."""
        pass

    def extract_to_markdown(self, pdf_path: str) -> str:
        """
        Extract text from PDF file and convert to Markdown.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Markdown string
        """
        md_text = pymupdf4llm.to_markdown(pdf_path)
        return md_text
