"""Main data ingestion pipeline for insurance claim documents."""

import os
import sys
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_ingestion.pdf_processor import PDFProcessor
from src.data_ingestion.structure_identifier import StructureIdentifier
from src.data_ingestion.chunker import HierarchicalChunker
from src.data_ingestion.indexer import HierarchicalIndexer
from src.utils.llm_utils import get_llm, get_embedding_model


class IngestionPipeline:
    """Main pipeline for ingesting insurance claim PDFs."""
    
    def __init__(
        self,
        config_path: str,
        persist_directory: str = "./data/vector_store",
        collection_name: str = "insurance_claims"
    ):
        """
        Initialize ingestion pipeline.
        
        Args:
            config_path: Path to config YAML file (required)
            persist_directory: Directory to persist vector store
            collection_name: Name of ChromaDB collection
            
        Raises:
            FileNotFoundError: If config file does not exist
        """
        # Load configuration
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.pdf_processor = PDFProcessor()
        self.structure_identifier = StructureIdentifier(llm=get_llm(
            model=self.config.get('llm', {}).get('model'),
            temperature=self.config.get('llm', {}).get('temperature')
        ))
        
        # Initialize chunker with config values
        chunking_config = self.config.get('chunking', {})
        self.chunker = HierarchicalChunker(
            small_chunk_size=chunking_config.get('small_chunk_size', 128),
            medium_chunk_size=chunking_config.get('medium_chunk_size', 256),
            large_chunk_size=chunking_config.get('large_chunk_size', 512),
            chunk_overlap=chunking_config.get('chunk_overlap', 50)
        )
        
        # Initialize indexer
        vector_store_config = self.config.get('vector_store', {})
        self.indexer = HierarchicalIndexer(
            persist_directory=persist_directory or vector_store_config.get('persist_directory', './data/vector_store'),
            collection_name=collection_name or vector_store_config.get('collection_chunks', 'chunks'),
            embedding_model=get_embedding_model(
                model=self.config.get('llm', {}).get('embedding_model', 'text-embedding-3-small')
            )
        )
    
    def process_pdf(
        self,
        pdf_path: str,
        claim_id: Optional[str] = None,
        document_metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Process a single PDF file through the ingestion pipeline.
        
        Args:
            pdf_path: Path to PDF file
            claim_id: Optional claim ID (if None, will be generated from filename)
            document_metadata: Optional additional document metadata
            
        Returns:
            Dictionary with processing results and statistics
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Generate claim_id from filename if not provided
        if claim_id is None:
            claim_id = pdf_path.stem
        
        print(f"Processing PDF: {pdf_path.name}")
        print(f"Claim ID: {claim_id}")
        
        # Step 1: Extract text from PDF
        print("Step 1: Extracting text from PDF...")
        pdf_data = self.pdf_processor.extract_text(str(pdf_path))
        print(f"  Extracted {len(pdf_data['text'])} characters from {pdf_data['metadata']['total_pages']} pages")
        
        # Step 2: Identify document structure
        print("Step 2: Identifying document structure...")
        structure = self.structure_identifier.identify_structure(
            text=pdf_data['text'],
            document_name=pdf_path.name
        )
        print(f"  Identified {len(structure.get('sections', []))} sections")
        print(f"  Structure: {structure}")

        return
        
        # Step 3: Process each section with hierarchical chunking
        print("Step 3: Creating hierarchical chunks...")
        all_chunks = {"small": [], "medium": [], "large": []}
        document_id = pdf_path.stem
        
        for section in structure.get('sections', []):
            section_name = section.get('section_name', 'Unknown')
            section_text = self.structure_identifier.extract_section_text(
                pdf_data['text'],
                section
            )
            
            if not section_text.strip():
                continue
            
            # Chunk the section
            section_chunks = self.chunker.chunk_section(
                section_text=section_text,
                section_name=section_name,
                claim_id=claim_id,
                document_id=document_id,
                section_metadata={
                    "section_type": section.get('section_type', 'narrative'),
                    "page_number": section.get('page_number', 1),
                    "description": section.get('description', '')
                }
            )
            
            # Aggregate chunks
            for level in ["small", "medium", "large"]:
                all_chunks[level].extend(section_chunks[level])
        
        print(f"  Created {len(all_chunks['small'])} small, {len(all_chunks['medium'])} medium, {len(all_chunks['large'])} large chunks")
        
        # Step 4: Index chunks into vector store
        print("Step 4: Indexing chunks into vector store...")
        metadata = {
            "file_name": pdf_path.name,
            "file_path": str(pdf_path),
            "processed_at": datetime.now().isoformat(),
            "total_pages": pdf_data['metadata']['total_pages'],
            **(document_metadata or {})
        }
        
        counts = self.indexer.index_chunks(
            chunks=all_chunks,
            claim_id=claim_id,
            document_id=document_id,
            metadata=metadata
        )
        
        print(f"  Indexed {counts['small']} small, {counts['medium']} medium, {counts['large']} large chunks")
        
        # Return processing results
        return {
            "claim_id": claim_id,
            "document_id": document_id,
            "file_name": pdf_path.name,
            "pages": pdf_data['metadata']['total_pages'],
            "sections": len(structure.get('sections', [])),
            "chunks_created": {
                "small": len(all_chunks['small']),
                "medium": len(all_chunks['medium']),
                "large": len(all_chunks['large'])
            },
            "chunks_indexed": counts,
            "structure": structure
        }
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the indexed data."""
        return self.indexer.get_collection_stats()


def main():
    """Main entry point for the ingestion pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Insurance Claim Data Ingestion Pipeline")
    parser.add_argument(
        "--pdf",
        type=str,
        required=True,
        help="Path to PDF file to process"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file (default: config/config.yaml)"
    )
    parser.add_argument(
        "--claim-id",
        type=str,
        help="Claim ID (optional, will use filename if not provided)"
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="./data/vector_store",
        help="Directory to persist vector store"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="insurance_claims",
        help="ChromaDB collection name"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = IngestionPipeline(
        config_path=args.config,
        persist_directory=args.persist_dir,
        collection_name=args.collection
    )
    
    # Process PDF
    result = pipeline.process_pdf(args.pdf, claim_id=args.claim_id)
    print("\n" + "="*50)
    print("Processing Complete!")
    print("="*50)
    print(f"Claim ID: {result['claim_id']}")
    print(f"Document: {result['file_name']}")
    print(f"Pages: {result['pages']}")
    print(f"Sections: {result['sections']}")
    print(f"Chunks Created: {result['chunks_created']}")
    print(f"Chunks Indexed: {result['chunks_indexed']}")
    
    # Print index statistics
    stats = pipeline.get_index_stats()
    print(f"\nIndex Statistics:")
    print(f"  Total chunks in collection: {stats['total_chunks']}")
    print(f"  Collection name: {stats['collection_name']}")
    print(f"  Persist directory: {stats['persist_directory']}")


if __name__ == "__main__":
    main()
