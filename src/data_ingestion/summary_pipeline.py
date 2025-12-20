"""Separate pipeline for generating summaries directly from PDFs using SentenceSplitter."""

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
from llama_index.core import Settings, Document
from llama_index.llms.openai import OpenAI

load_dotenv()


class SummaryPipeline:
    """Pipeline for generating summaries directly from PDFs using SentenceSplitter."""
    
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
        
        # Initialize ChromaDB manager for summaries
        self.summary_manager = ChromaDBManager(
            persist_directory=self.vector_store_config['persist_directory'],
            collection_name=self.vector_store_config['collection_summaries'],
            embedding_model_name=self.vector_store_config['embedding_model']
        )
        
        # Initialize summary generator
        self.summary_generator = SummaryGenerator(
            chunk_size=self.summarization_config['chunk_size'],
            chunk_overlap=self.summarization_config['chunk_overlap']
        )
    
    def execute(
        self,
        text: str,
        claim_id: str,
        source_file: str,
        source_path: str
    ):
        """
        Execute summary generation pipeline on text using MapReduce with SentenceSplitter.
        
        Args:
            text: Text to process
            claim_id: Claim ID for metadata
            source_file: Source filename for metadata
            source_path: Source file path for metadata
        
        Returns:
            Dictionary with 'summaries_generated', 'summary_ids', and 'claim_id'
        """
        print("=" * 60)
        print("Summary Generation Pipeline (MapReduce)")
        print("=" * 60)
        print(f"Source file: {source_file}")
        print(f"Claim ID: {claim_id}")
        print(f"Text length: {len(text)} characters")
        print("-" * 60)
        
        # Step 1: Create document with metadata
        print("\nStep 1: Creating document...")
        doc = Document(
            text=text,
            metadata={
                'claim_id': claim_id,
                'source_file': source_file,
                'source_path': source_path
            }
        )
        
        # Step 2: Generate summaries using MapReduce
        print(f"\nStep 2: Generating summaries (MapReduce)...")
        print(f"  Chunk size: {self.summarization_config['chunk_size']} tokens")
        print(f"  Chunk overlap: {self.summarization_config['chunk_overlap']} tokens")
        
        try:
            # Use SentenceSplitter to chunk and summarize
            summary_result = self.summary_generator.generate_summaries_from_document(
                doc,
                show_progress=True
            )
            
            # Combine all summaries
            all_summaries = (
                summary_result['chunk'] + 
                summary_result['document']
            )
            
            print(f"  Generated {len(summary_result['chunk'])} chunk-level summaries")
            print(f"  Generated {len(summary_result['document'])} document-level summaries")
            print(f"  Total: {len(all_summaries)} summaries")
            
            summary_nodes = all_summaries
            
            # Validate summaries
            summaries_with_errors = [s for s in summary_nodes if s.metadata.get('summary_error')]
            if summaries_with_errors:
                print(f"  Warning: {len(summaries_with_errors)} summaries had errors and used fallback text")
        except Exception as e:
            print(f"  Error generating summaries: {e}")
            raise
        
        # Step 3: Store summaries in ChromaDB
        print("\nStep 3: Storing summaries in ChromaDB...")
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
            'summaries_generated': len(summary_nodes),
            'summary_ids': summary_ids,
            'claim_id': claim_id
        }
    
    def print_summary_stats(self):
        """Print statistics about summaries in ChromaDB."""
        print("=" * 60)
        print("ChromaDB Summary Statistics")
        print("=" * 60)
        
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
                chunk_count = 0
                doc_count = 0
                
                for metadata in all_summaries['metadatas']:
                    if metadata:
                        if 'claim_id' in metadata:
                            claim_ids.add(metadata['claim_id'])
                        if metadata.get('summary_level') == 'chunk':
                            chunk_count += 1
                        elif metadata.get('summary_level') == 'document':
                            doc_count += 1
                
                print(f"  Chunk summaries: {chunk_count}")
                print(f"  Document summaries: {doc_count}")
                print(f"  Unique claim IDs: {len(claim_ids)}")
                if claim_ids:
                    print(f"  Claim IDs: {', '.join(sorted(claim_ids))}")
        except Exception as e:
            print(f"\nError accessing summaries: {e}")
        
        print("\n" + "=" * 60)

