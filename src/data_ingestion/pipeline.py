"""Main data ingestion pipeline orchestrator that coordinates summary and hierarchical pipelines."""

import sys
from pathlib import Path
from typing import Optional, Dict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_ingestion.pdf_processor import PDFProcessor
from src.data_ingestion.summary_pipeline import SummaryPipeline
from src.data_ingestion.hierarchical_pipeline import HierarchicalPipeline
from src.utils.config_loader import load_config


class IngestionPipeline:
    """Main orchestrator pipeline that coordinates summary and hierarchical chunking pipelines."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize ingestion pipeline orchestrator.
        
        Flow:
        1. Initialize
        2. Process PDF (extract text)
        3. Execute Hierarchical pipeline
        4. Execute Summary pipeline
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Initialize PDF processor for text extraction
        self.pdf_processor = PDFProcessor()
        
        # Initialize sub-pipelines
        self.summary_pipeline = SummaryPipeline(config_path)
        self.hierarchical_pipeline = HierarchicalPipeline(config_path)

    def process_pdf(
        self,
        pdf_path: str,
        claim_id: Optional[str] = None,
        generate_chunks: bool = True,
        generate_summaries: bool = True,
        split_into_sections: bool = False
    ):
        """
        Process a single PDF file using both summary and hierarchical pipelines.
        
        Flow:
        1. Initialize (already done in __init__)
        2. Process PDF (extract text)
        3. Execute Hierarchical pipeline
        4. Execute Summary pipeline
        
        Args:
            pdf_path: Path to PDF file
            claim_id: Optional claim ID (if None, will be generated from filename)
            split_into_sections: If True, parse markdown into sections before hierarchical chunking
            generate_summaries: If True, generate summaries using SummaryPipeline
            generate_chunks: If True, generate hierarchical chunks using HierarchicalPipeline
        
        Returns:
            Dictionary with results from both pipelines
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Generate claim_id from filename if not provided
        if claim_id is None:
            claim_id = pdf_path.stem
        
        print("=" * 60)
        print("Data Ingestion Pipeline Orchestrator")
        print("=" * 60)
        print(f"PDF: {pdf_path.name}")
        print(f"Claim ID: {claim_id}")
        print(f"Generate summaries: {generate_summaries}")
        print(f"Generate chunks: {generate_chunks}")
        print(f"Split into sections: {split_into_sections}")
        print("=" * 60)
        
        results = {
            'claim_id': claim_id,
            'summary_results': None,
            'hierarchical_results': None
        }
        
        # Step 2: Process PDF - Extract text from PDF
        print("\n" + "=" * 60)
        print("Step 2: Processing PDF - Extracting text...")
        print("=" * 60)
        md_text = self.pdf_processor.extract_to_markdown(str(pdf_path))
        print(f"  Extracted {len(md_text)} characters from PDF")
        
        source_file = pdf_path.name
        source_path = str(pdf_path)
        
        # Step 3: Execute Hierarchical pipeline
        if generate_chunks:
            print("\n" + "=" * 60)
            print("Step 3: Executing Hierarchical Pipeline...")
            print("=" * 60)
            try:
                hierarchical_results = self.hierarchical_pipeline.execute(
                    text=md_text,
                    claim_id=claim_id,
                    source_file=source_file,
                    source_path=source_path,
                    split_into_sections=split_into_sections
                )
                results['hierarchical_results'] = hierarchical_results
            except Exception as e:
                print(f"Error in hierarchical pipeline: {e}")
                if generate_summaries:
                    print("Continuing with summary pipeline...")
                else:
                    raise
        
        # Step 4: Execute Summary pipeline
        if generate_summaries:
            print("\n" + "=" * 60)
            print("Step 4: Executing Summary Pipeline...")
            print("=" * 60)
            try:
                summary_results = self.summary_pipeline.execute(
                    text=md_text,
                    claim_id=claim_id,
                    source_file=source_file,
                    source_path=source_path
                )
                results['summary_results'] = summary_results
            except Exception as e:
                print(f"Error in summary pipeline: {e}")
                if generate_chunks:
                    print("Hierarchical pipeline completed successfully.")
                raise
        
        print("\n" + "=" * 60)
        print("Pipeline Orchestration Complete!")
        print("=" * 60)
        
        return results

    def print_existing_chunks(self):
        """Retrieve and print all existing chunks from ChromaDB for debugging."""
        self.hierarchical_pipeline.print_existing_chunks()
    
    def print_collections_info(self):
        """Print information about all ChromaDB collections."""
        print("=" * 60)
        print("ChromaDB Collections Information")
        print("=" * 60)
        
        try:
            # Get chroma_client from hierarchical pipeline (both pipelines use the same client)
            chroma_client = self.hierarchical_pipeline.chroma_manager.chroma_client
            
            # List all collections
            collections = chroma_client.list_collections()
            
            if not collections:
                print("\nNo collections found in ChromaDB.")
                return
            
            print(f"\nFound {len(collections)} collection(s):\n")
            
            # Print header
            print(f"{'Collection Name':<30} {'Collection ID':<40} {'Items':<10}")
            print("-" * 80)
            
            # Print each collection's information
            for collection in collections:
                collection_name = collection.name
                collection_id = str(collection.id)  # Convert UUID to string
                try:
                    # Get collection count
                    collection_obj = chroma_client.get_collection(name=collection_name)
                    item_count = collection_obj.count()
                except Exception as e:
                    item_count = f"Error: {e}"
                
                print(f"{collection_name:<30} {collection_id:<40} {item_count:<10}")
            
            print("-" * 80)
            print(f"\nTotal collections: {len(collections)}")
            
            # Print docstore information
            self._print_docstore_info()
            
        except Exception as e:
            print(f"\nError retrieving collections information: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 60)
    
    def _print_docstore_info(self):
        """Print information about the docstore."""
        try:
            persist_dir = Path(self.config['vector_store']['persist_directory'])
            docstore_path = persist_dir / "docstore.json"
            
            print("\n" + "=" * 60)
            print("Docstore Information")
            print("=" * 60)
            
            # Check if docstore file exists
            if not docstore_path.exists():
                print("\nDocstore file not found.")
                print(f"Expected location: {docstore_path}")
                return
            
            # Get file size
            file_size = docstore_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            
            print(f"\nDocstore file: {docstore_path}")
            print(f"File size: {file_size:,} bytes ({file_size_mb:.2f} MB)")
            
            # Try to access docstore from hierarchical pipeline
            try:
                docstore = self.hierarchical_pipeline.chroma_manager.index.storage_context.docstore
                
                if hasattr(docstore, 'docs') and docstore.docs:
                    total_nodes = len(docstore.docs)
                    print(f"Total nodes: {total_nodes}")
                    
                    # Get some statistics about nodes
                    from llama_index.core.node_parser import get_leaf_nodes
                    all_nodes = [docstore.get_document(node_id) for node_id in list(docstore.docs.keys())[:1000]]  # Sample first 1000 for performance
                    if len(docstore.docs) <= 1000:
                        all_nodes = [docstore.get_document(node_id) for node_id in docstore.docs.keys()]
                    
                    leaf_nodes = get_leaf_nodes(all_nodes)
                    leaf_count = len(leaf_nodes)
                    parent_count = total_nodes - leaf_count if len(all_nodes) == total_nodes else "N/A"
                    
                    print(f"Leaf nodes: {leaf_count}")
                    if parent_count != "N/A":
                        print(f"Parent nodes: {parent_count}")
                    
                    # Get claim_id distribution
                    claim_ids = set()
                    for node in all_nodes[:100]:  # Sample for performance
                        if hasattr(node, 'metadata') and node.metadata:
                            claim_id = node.metadata.get('claim_id')
                            if claim_id:
                                claim_ids.add(claim_id)
                    
                    if claim_ids:
                        print(f"Unique claim IDs (sampled): {len(claim_ids)}")
                        print(f"Claim IDs: {', '.join(sorted(claim_ids))}")
                    
                else:
                    print("Docstore is empty or not accessible.")
                    
            except Exception as e:
                print(f"Error accessing docstore: {e}")
                print("File exists but could not load docstore content.")
            
        except Exception as e:
            print(f"\nError retrieving docstore information: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point for the ingestion pipeline orchestrator."""
    import argparse

    parser = argparse.ArgumentParser(description="PDF Data Ingestion Pipeline Orchestrator")
    parser.add_argument(
        "--pdf",
        type=str,
        help="Path to PDF file to process (required unless --print-existing-chunks is used)"
    )
    parser.add_argument(
        "--claim-id",
        type=str,
        help="Claim ID (optional, will use filename if not provided)"
    )
    parser.add_argument(
        "--print-existing-chunks",
        action="store_true",
        help="Print all existing chunks from ChromaDB without processing any PDF (debugging mode)"
    )
    parser.add_argument(
        "--print-store-info",
        action="store_true",
        help="Print information about vector store (ChromaDB collections and docstore)"
    )
    parser.add_argument(
        "--split-into-sections",
        action="store_true",
        help="Split markdown into sections before chunking (default: use raw markdown directly)"
    )
    parser.add_argument(
        "--no-summaries",
        action="store_true",
        help="Skip summary generation (only run hierarchical chunking)"
    )
    parser.add_argument(
        "--no-chunks",
        action="store_true",
        help="Skip hierarchical chunking (only run summary generation)"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = IngestionPipeline()
    
    # If print-store-info flag is set, print store info and exit
    if args.print_store_info:
        pipeline.print_collections_info()
        return
    
    # If print-existing-chunks flag is set, skip PDF processing
    if args.print_existing_chunks:
        pipeline.print_existing_chunks()
        return
    
    # Otherwise, require PDF argument
    if not args.pdf:
        parser.error("--pdf is required unless --print-existing-chunks or --print-store-info is used")
    
    # Process PDF
    pipeline.process_pdf(
        pdf_path=args.pdf,
        claim_id=args.claim_id,
        generate_chunks=not args.no_chunks,
        generate_summaries=not args.no_summaries,
        split_into_sections=args.split_into_sections
    )


if __name__ == "__main__":
    main()
