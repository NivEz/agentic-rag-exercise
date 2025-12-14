"""PDF processing module for extracting text from insurance claim documents."""

from typing import Dict
from pathlib import Path
import pymupdf


class PDFProcessor:
    """Processes PDF files and extracts text."""
    
    def __init__(self):
        """Initialize PDF processor."""
        pass
    
    def extract_text(self, pdf_path: str) -> Dict[str, any]:
        """
        Extract text from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary containing:
                - text: Full extracted text
                - pages: List of page texts with page numbers
                - metadata: Simple metadata with total_pages only
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        full_text = []
        pages = []
        
        # Open PDF with PyMuPDF
        doc = pymupdf.open(str(pdf_path))
        
        try:
            # Extract text from each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                
                if page_text.strip():
                    full_text.append(page_text)
                    pages.append({
                        "page_number": page_num + 1,
                        "text": page_text,
                        "char_count": len(page_text)
                    })
        finally:
            doc.close()
        
        return {
            "text": "\n\n".join(full_text),
            "pages": pages,
            "metadata": {
                "total_pages": len(pages)
            },
            "file_path": str(pdf_path),
            "file_name": pdf_path.name
        }
