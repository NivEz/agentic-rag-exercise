"""Separate pipeline for generating summaries from existing chunks in ChromaDB."""

import sys
from pathlib import Path
from typing import Optional, List
import os
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_ingestion.summary_generator import SummaryGenerator
from src.utils.config_loader import load_config, get_vector_store_config, get_summarization_config, get_llm_config
from src.utils.vector_store import ChromaDBManager
from llama_index.core import Settings
from llama_index.core.node_parser import get_leaf_nodes
from llama_index.llms.openai import OpenAI

load_dotenv()


class SummaryPipeline:
    """Pipeline for generating summaries from existing chunks in ChromaDB."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize summary pipeline."""
        # Load configuration
        self.config = load_config(config_path)
        self.vector_store_config = get_vector_store_config(self.config)
        self.summarization_config = get_summarization_config(self.config)
        self.llm_config = get_llm_config(self.config)
        
        # Initialize LLM in Settings
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        Settings.llm = OpenAI(
            model=self.llm_config['model'],
            temperature=self.llm_config['temperature'],
            api_key=openai_api_key
        )
        
        # Initialize ChromaDB manager for chunks (read-only)
        self.chunks_manager = ChromaDBManager(
            persist_directory=self.vector_store_config['persist_directory'],
            collection_name=self.vector_store_config['collection_chunks'],
            embedding_model_name=self.vector_store_config['embedding_model']
        )
        
        # Initialize ChromaDB manager for summaries
        self.summary_manager = ChromaDBManager(
            persist_directory=self.vector_store_config['persist_directory'],
            collection_name=self.vector_store_config['collection_summaries'],
            embedding_model_name=self.vector_store_config['embedding_model']
        )
        
        # Initialize summary generator
        self.summary_generator = SummaryGenerator(
            summary_instruction=self.summarization_config['summary_instruction']
        )
    
    def get_chunks_by_claim_id(self, claim_id: Optional[str] = None) -> List:
        """
        Retrieve chunks from ChromaDB, optionally filtered by claim_id.
        
        Args:
            claim_id: Optional claim ID to filter chunks. If None, retrieves all chunks.
            
        Returns:
            List of chunk nodes
        """
        print(f"\nRetrieving chunks from ChromaDB...")
        
        try:
            # Access the docstore from the index
            docstore = self.chunks_manager.index.storage_context.docstore
            
            if not hasattr(docstore, 'docs') or not docstore.docs:
                print("  No chunks found in ChromaDB.")
                return []
            
            # Get all nodes from docstore
            all_node_ids = list(docstore.docs.keys())
            all_chunks = [docstore.get_document(node_id) for node_id in all_node_ids]
            
            # Filter by claim_id if provided
            if claim_id:
                filtered_chunks = [
                    chunk for chunk in all_chunks 
                    if hasattr(chunk, 'metadata') 
                    and chunk.metadata 
                    and chunk.metadata.get('claim_id') == claim_id
                ]
                print(f"  Found {len(filtered_chunks)} chunks for claim_id: {claim_id}")
                return filtered_chunks
            else:
                print(f"  Found {len(all_chunks)} total chunks")
                return all_chunks
                
        except Exception as e:
            print(f"  Error retrieving chunks: {e}")
            raise
    
    def generate_summaries_for_chunks(
        self,
        claim_id: Optional[str] = None
    ):
        """
        Generate summaries for chunks in ChromaDB.
        
        Args:
            claim_id: Optional claim ID to filter chunks. If None, processes all chunks.
        """
        print("=" * 60)
        print("Summary Generation Pipeline")
        print("=" * 60)
        print(f"Claim ID filter: {claim_id if claim_id else 'None (all chunks)'}")
        print("-" * 60)
        
        # Step 1: Retrieve chunks
        print("\nStep 1: Retrieving chunks from ChromaDB...")
        chunks = self.get_chunks_by_claim_id(claim_id)
        
        if not chunks:
            print("No chunks found. Please run the ingestion pipeline first.")
            return {
                'chunks_processed': 0,
                'summaries_generated': 0,
                'summary_ids': []
            }
        
        # Step 2: Get leaf nodes (smallest chunks for summarization)
        print("\nStep 2: Extracting leaf nodes for summarization...")
        leaf_chunks = get_leaf_nodes(chunks)
        print(f"  Found {len(leaf_chunks)} leaf chunks out of {len(chunks)} total chunks")
        
        if not leaf_chunks:
            print("  No leaf chunks found for summarization.")
            return {
                'chunks_processed': len(chunks),
                'summaries_generated': 0,
                'summary_ids': []
            }
        
        # Step 3: Generate summaries
        print("\nStep 3: Generating summaries...")
        try:
            summary_nodes = self.summary_generator.generate_summaries(
                leaf_chunks, 
                show_progress=True
            )
            print(f"  Successfully generated {len(summary_nodes)} summaries")
            
            # Validate summaries
            summaries_with_errors = [s for s in summary_nodes if s.metadata.get('summary_error')]
            if summaries_with_errors:
                print(f"  Warning: {len(summaries_with_errors)} summaries had errors and used fallback text")
        except Exception as e:
            print(f"  Error generating summaries: {e}")
            raise
        
        # Step 4: Store summaries in ChromaDB
        print("\nStep 4: Storing summaries in ChromaDB...")
        try:
            summary_ids = self.summary_manager.add_nodes(summary_nodes)
            print(f"  Successfully stored {len(summary_ids)} summaries")
            print(f"  Collection: {self.vector_store_config['collection_summaries']}")
            print(f"  Persist directory: {self.vector_store_config['persist_directory']}")
        except Exception as e:
            print(f"  Error storing summaries: {e}")
            raise
        
        print("\n" + "=" * 60)
        print("Summary Generation Complete!")
        print("=" * 60)
        
        return {
            'chunks_processed': len(chunks),
            'leaf_chunks': len(leaf_chunks),
            'summaries_generated': len(summary_nodes),
            'summary_ids': summary_ids
        }
    
    def print_summary_stats(self):
        """Print statistics about chunks and summaries in ChromaDB."""
        print("=" * 60)
        print("ChromaDB Summary Statistics")
        print("=" * 60)
        
        # Get chunk count
        try:
            chunk_docstore = self.chunks_manager.index.storage_context.docstore
            chunk_count = len(chunk_docstore.docs) if hasattr(chunk_docstore, 'docs') else 0
            
            # Count leaf chunks
            if chunk_count > 0:
                all_chunks = [chunk_docstore.get_document(node_id) for node_id in chunk_docstore.docs.keys()]
                leaf_chunks = get_leaf_nodes(all_chunks)
                leaf_count = len(leaf_chunks)
            else:
                leaf_count = 0
            
            print(f"\nChunks Collection:")
            print(f"  Total chunks: {chunk_count}")
            print(f"  Leaf chunks: {leaf_count}")
        except Exception as e:
            print(f"\nError accessing chunks: {e}")
        
        # Get summary count from ChromaDB collection directly
        try:
            # Get the actual ChromaDB collection
            summary_collection = self.summary_manager.chroma_client.get_collection(
                name=self.vector_store_config['collection_summaries']
            )
            summary_count = summary_collection.count()
            
            print(f"\nSummaries Collection:")
            print(f"  Total summaries: {summary_count}")
            
            # Get claim_id distribution
            if summary_count > 0:
                # Get all summaries metadata
                all_summaries = summary_collection.get(include=['metadatas'])
                claim_ids = set()
                for metadata in all_summaries['metadatas']:
                    if metadata and 'claim_id' in metadata:
                        claim_ids.add(metadata['claim_id'])
                
                print(f"  Unique claim IDs: {len(claim_ids)}")
                if claim_ids:
                    print(f"  Claim IDs: {', '.join(sorted(claim_ids))}")
        except Exception as e:
            print(f"\nError accessing summaries: {e}")
        
        print("\n" + "=" * 60)


def main():
    """Main entry point for the summary pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Summary Generation Pipeline - Generate summaries from existing chunks")
    parser.add_argument(
        "--claim-id",
        type=str,
        help="Claim ID to filter chunks (optional, processes all chunks if not provided)"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print statistics about chunks and summaries without generating new ones"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = SummaryPipeline()
    
    # If stats flag is set, just print stats
    if args.stats:
        pipeline.print_summary_stats()
        return
    
    # Otherwise, generate summaries
    result = pipeline.generate_summaries_for_chunks(
        claim_id=args.claim_id
    )
    
    print(f"\nResults:")
    print(f"  Chunks processed: {result.get('chunks_processed', 0)}")
    print(f"  Summaries generated: {result.get('summaries_generated', 0)}")


if __name__ == "__main__":
    main()

