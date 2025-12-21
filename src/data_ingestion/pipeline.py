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
        3. Execute Hierarchical pipeline (if generate_chunks=True)
        4. Execute Summary pipeline (if generate_summaries=True)
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
        plain_text = self.pdf_processor.extract_to_text(str(pdf_path))
        print(f"  Extracted {len(plain_text)} characters from PDF")
        
        source_file = pdf_path.name
        source_path = str(pdf_path)
        
        # Step 3: Execute Hierarchical pipeline
        if generate_chunks:
            print("\n" + "=" * 60)
            print("Step 3: Executing Hierarchical Pipeline...")
            print("=" * 60)
            try:
                hierarchical_results = self.hierarchical_pipeline.execute(
                    text=plain_text,
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
                    text=plain_text,
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
            
            # Calculate total items as sum of items in all collections
            try:
                chroma_client = self.hierarchical_pipeline.chroma_manager.chroma_client
                collections = chroma_client.list_collections()
                
                total_items = 0
                for collection in collections:
                    try:
                        collection_obj = chroma_client.get_collection(name=collection.name)
                        total_items += collection_obj.count()
                    except Exception:
                        pass
                
                print(f"Total items: {total_items} (sum of all collections)")
                
                # Get docstore node count and parent nodes
                from llama_index.core import StorageContext
                from llama_index.core.node_parser import get_leaf_nodes
                
                storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
                docstore = storage_context.docstore
                
                if hasattr(docstore, 'docs') and docstore.docs:
                    docstore_nodes = len(docstore.docs)
                    print(f"Docstore nodes: {docstore_nodes}")
                    
                    # Calculate parent nodes (non-leaf nodes)
                    all_nodes = [docstore.get_document(node_id) for node_id in docstore.docs.keys()]
                    leaf_nodes = get_leaf_nodes(all_nodes)
                    parent_nodes = docstore_nodes - len(leaf_nodes)
                    print(f"Parent nodes: {parent_nodes}")
                else:
                    print("Docstore nodes: 0")
                    print("Parent nodes: 0")
                    
            except Exception as e:
                print(f"Error calculating items: {e}")
            
        except Exception as e:
            print(f"\nError retrieving docstore information: {e}")


def main():
    """Main entry point for the ingestion pipeline orchestrator."""
    import argparse

    parser = argparse.ArgumentParser(
        description="PDF Data Ingestion Pipeline Orchestrator. "
                    "Use --chunks to run only chunking, --summaries to run only summaries, "
                    "or neither to run both."
    )
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
    
    # Create mutually exclusive group for pipeline selection
    # Required only when processing PDFs (not for debug flags)
    pipeline_group = parser.add_mutually_exclusive_group(required=False)
    pipeline_group.add_argument(
        "--chunks",
        action="store_true",
        help="Run only hierarchical chunking pipeline"
    )
    pipeline_group.add_argument(
        "--summaries",
        action="store_true",
        help="Run only summary generation pipeline"
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
    
    # Validate that exactly one pipeline is selected (required when processing PDFs)
    if not args.chunks and not args.summaries:
        parser.error("Either --chunks or --summaries must be specified when processing a PDF.")
    
    # Determine which pipeline to run (mutually exclusive)
    generate_chunks = args.chunks
    generate_summaries = args.summaries
    
    # Process PDF
    pipeline.process_pdf(
        pdf_path=args.pdf,
        claim_id=args.claim_id,
        generate_chunks=generate_chunks,
        generate_summaries=generate_summaries,
        split_into_sections=args.split_into_sections
    )


if __name__ == "__main__":
    main()
