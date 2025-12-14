"""Structure identification module for identifying document sections using LLM."""

import json
from typing import Dict, List, Optional
from llama_index.core.llms import ChatMessage
from src.utils.llm_utils import get_llm


class StructureIdentifier:
    """Identifies document structure (sections, subsections) using LLM."""
    
    def __init__(self, llm=None):
        """
        Initialize structure identifier.
        
        Args:
            llm: LLM instance (if None, will create one)
        """
        self.llm = llm or get_llm()
    
    def identify_structure(self, text: str, document_name: str = "") -> Dict[str, any]:
        """
        Identify document structure using LLM.
        
        Args:
            text: Full document text
            document_name: Name of the document (optional)
            
        Returns:
            Dictionary containing:
                - sections: List of sections with names and boundaries
                - structure_type: Type of document structure identified
        """
        # Truncate text if too long (to fit in context window)
        max_chars = 20000  # Leave room for prompt and response
        truncated_text = text[:max_chars] if len(text) > max_chars else text
        
        prompt = self._create_structure_prompt(truncated_text, document_name)
        
        # Use LLM to identify structure
        response = self.llm.complete(prompt)
        structure_json = self._parse_response(response.text)
        
        return structure_json
    
    def _create_structure_prompt(self, text: str, document_name: str) -> str:
        """Create prompt for structure identification."""
        return f"""You are analyzing an insurance claim document to identify its structure.

Document Name: {document_name}

Document Text:
{text}

Please identify the document structure and return a JSON object with the following format:
{{
    "structure_type": "insurance_claim",
    "sections": [
        {{
            "section_name": "Section Name",
            "section_type": "header" | "subsection" | "table" | "narrative",
            "start_char": 0,
            "end_char": 1000,
            "page_number": 1,
            "description": "Brief description of section content"
        }}
    ]
}}

Guidelines:
1. Identify all major sections (e.g., "Initial Report", "Medical Records", "Claim Details", "Timeline", etc.)
2. For each section, provide start and end character positions in the original text
3. Include page numbers if available
4. Describe the content type of each section
5. Ensure sections don't overlap
6. Cover the entire document

Return only valid JSON, no additional text."""

    def _parse_response(self, response_text: str) -> Dict[str, any]:
        """Parse LLM response into structured format."""
        # Try to extract JSON from response
        response_text = response_text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
        
        try:
            structure = json.loads(response_text)
            return structure
        except json.JSONDecodeError as e:
            # Fallback: return default structure
            print(f"Warning: Failed to parse structure JSON: {e}")
            return {
                "structure_type": "unknown",
                "sections": [
                    {
                        "section_name": "Full Document",
                        "section_type": "narrative",
                        "start_char": 0,
                        "end_char": -1,
                        "page_number": 1,
                        "description": "Entire document treated as single section"
                    }
                ]
            }
    
    def extract_section_text(self, full_text: str, section: Dict[str, any]) -> str:
        """
        Extract text for a specific section.
        
        Args:
            full_text: Full document text
            section: Section dictionary with start_char and end_char
            
        Returns:
            Section text
        """
        start = section.get("start_char", 0)
        end = section.get("end_char", len(full_text))
        
        if end == -1:
            end = len(full_text)
        
        return full_text[start:end]
