"""PDF processing module for extracting text from insurance claim documents."""

from typing import Dict
from pathlib import Path
import pymupdf.layout
import pymupdf4llm
import pymupdf


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
    
    def extract_to_text(self, pdf_path: str) -> str:
        """
        Extract plain text from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Plain text string
        """
        doc = pymupdf.open(pdf_path)
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n".join(text_parts)
