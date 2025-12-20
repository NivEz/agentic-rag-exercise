"""Separate pipeline for generating summaries directly from PDFs using SentenceSplitter."""

import sys
from pathlib import Path
from typing import Optional, List
import os
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_ingestion.summary_generator import SummaryGenerator
from src.data_ingestion.pdf_processor import PDFProcessor
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
        
        # Initialize PDF processor
        self.pdf_processor = PDFProcessor()
        
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
    
    def generate_summaries_for_pdf(
        self,
        pdf_path: str,
        claim_id: Optional[str] = None
    ):
        """
        Generate summaries for a PDF using MapReduce with SentenceSplitter.
        
        Args:
            pdf_path: Path to PDF file
            claim_id: Optional claim ID (if None, will use filename)
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Generate claim_id from filename if not provided
        if claim_id is None:
            claim_id = pdf_path.stem
        
        print("=" * 60)
        print("Summary Generation Pipeline (MapReduce)")
        print("=" * 60)
        print(f"PDF: {pdf_path.name}")
        print(f"Claim ID: {claim_id}")
        print("-" * 60)
        
        # Step 1: Extract text from PDF
        print("\nStep 1: Extracting text from PDF...")
        md_text = self.pdf_processor.extract_to_markdown(str(pdf_path))
        print(f"  Extracted {len(md_text)} characters")
        
        # Step 2: Create document with metadata
        print("\nStep 2: Creating document...")
        doc = Document(
            text=md_text,
            metadata={
                'claim_id': claim_id,
                'source_file': pdf_path.name,
                'source_path': str(pdf_path)
            }
        )
        
        # Step 3: Generate summaries using MapReduce
        print(f"\nStep 3: Generating summaries (MapReduce)...")
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


def main():
    """Main entry point for the summary pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Summary Generation Pipeline - Generate summaries from PDFs using MapReduce")
    parser.add_argument(
        "--pdf",
        type=str,
        help="Path to PDF file to process"
    )
    parser.add_argument(
        "--claim-id",
        type=str,
        help="Claim ID (optional, will use filename if not provided)"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print statistics about summaries without generating new ones"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = SummaryPipeline()
    
    # If stats flag is set, just print stats
    if args.stats:
        pipeline.print_summary_stats()
        return
    
    # Otherwise, require PDF argument
    if not args.pdf:
        parser.error("--pdf is required unless --stats is used")
    
    # Generate summaries
    result = pipeline.generate_summaries_for_pdf(
        pdf_path=args.pdf,
        claim_id=args.claim_id
    )
    
    print(f"\nResults:")
    print(f"  Claim ID: {result.get('claim_id')}")
    print(f"  Summaries generated: {result.get('summaries_generated', 0)}")


if __name__ == "__main__":
    main()

