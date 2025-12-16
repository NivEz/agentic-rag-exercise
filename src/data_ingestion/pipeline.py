"""Main data ingestion pipeline using llama_index MarkdownNodeParser."""

import sys
from pathlib import Path
from typing import Optional, List, Dict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_ingestion.pdf_processor import PDFProcessor
from src.utils.config_loader import load_config, get_chunking_config, get_vector_store_config
from src.utils.vector_store import ChromaDBManager
from llama_index.core.node_parser import (
    MarkdownNodeParser,
    HierarchicalNodeParser,
    get_leaf_nodes
)
from llama_index.core import Document
from llama_index.core.schema import TextNode, BaseNode


class IngestionPipeline:
    """Main pipeline for ingesting PDFs using MarkdownNodeParser."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize ingestion pipeline."""
        self.pdf_processor = PDFProcessor()
        self.parser = MarkdownNodeParser()
        
        # Load configuration
        self.config = load_config(config_path)
        self.chunking_config = get_chunking_config(self.config)
        self.vector_store_config = get_vector_store_config(self.config)
        
        # Initialize HierarchicalNodeParser with chunk sizes (largest to smallest)
        # This creates a hierarchy: large -> medium -> small chunks
        chunk_sizes = [
            self.chunking_config['large_chunk_size'],
            self.chunking_config['medium_chunk_size'],
            self.chunking_config['small_chunk_size']
        ]

        self.hierarchical_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=chunk_sizes,
            chunk_overlap=self.chunking_config['chunk_overlap']
        )
        
        # Initialize ChromaDB manager for chunks
        self.chroma_manager = ChromaDBManager(
            persist_directory=self.vector_store_config['persist_directory'],
            collection_name=self.vector_store_config['collection_chunks'],
            embedding_model_name=self.vector_store_config['embedding_model']
        )

    def merge_tiny_nodes(self, nodes, min_chars=100):
        """
        Merges nodes smaller than `min_chars` into the NEXT node.
        """
        merged_nodes = []
        buffer_node = None

        for i, node in enumerate(nodes):
            # If we have a buffer from the previous step, merge current node into it
            if buffer_node:
                # Combine text
                new_text = buffer_node.get_content() + "\n\n" + node.get_content()
                node.text = new_text
                
                # Combine metadata (optional: prefer the deeper/more specific header)
                # usually the next node's metadata is fine to keep, or you can merge dicts
                node.metadata = {**buffer_node.metadata, **node.metadata}

                # Clear buffer
                buffer_node = None

            # Check if the (possibly merged) node is STILL too small
            if len(node.get_content()) < min_chars:
                # If this is the LAST node, we can't merge forward. 
                # We must merge backward to the previous node in `merged_nodes`
                if i == len(nodes) - 1 and merged_nodes:
                    prev_node = merged_nodes[-1]
                    prev_node.text += "\n\n" + node.get_content()
                else:
                    # Mark this node to be merged into the NEXT one
                    buffer_node = node
            else:
                # Node is big enough, keep it
                merged_nodes.append(node)
                
        return merged_nodes

    def chunk_sections(self, sections: List[BaseNode]) -> List[BaseNode]:
        """
        Chunk a list of sections at multiple granularity levels using HierarchicalNodeParser.
        """
        section_chunks = []
        for section in sections:
            section_chunks.extend(self.chunk_section_multi_granularity(section))
        return section_chunks

    def chunk_section_multi_granularity(self, node: BaseNode) -> Dict[str, List[BaseNode]]:
        """
        Chunk a single section/node at multiple granularity levels using HierarchicalNodeParser.
        
        Returns a dictionary with keys: 'small', 'medium', 'large'
        Each value is a list of chunks at that granularity level.
        """
        section_text = node.get_content()
        
        section_metadata = node.metadata.copy()
        
        # Create Document from section text
        section_doc = Document(text=section_text, metadata=section_metadata)
        
        # Use HierarchicalNodeParser to create hierarchical chunks
        hierarchical_nodes = self.hierarchical_parser.get_nodes_from_documents([section_doc])
        # print(f"Hierarchical nodes: {hierarchical_nodes}")

        return hierarchical_nodes

    def process_pdf(
        self,
        pdf_path: str,
        claim_id: Optional[str] = None
    ):
        """
        Process a single PDF file: convert to markdown and parse into nodes.
        
        Args:
            pdf_path: Path to PDF file
            claim_id: Optional claim ID (if None, will be generated from filename)
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Generate claim_id from filename if not provided
        if claim_id is None:
            claim_id = pdf_path.stem
        
        print(f"Processing PDF: {pdf_path.name}")
        print(f"Claim ID: {claim_id}")
        print("-" * 50)

        # Step 1: Extract text from PDF and convert to Markdown
        print("\nStep 1: Extracting text from PDF and converting to Markdown...")
        md_text = self.pdf_processor.extract_to_markdown(str(pdf_path))
        print(f"  Extracted {len(md_text)} characters")

        # Step 2: Create Document and parse into nodes
        print("\nStep 2: Parsing markdown into nodes...")
        doc = Document(text=md_text)
        section_nodes = self.parser.get_nodes_from_documents([doc])
        print(f"  Created {len(section_nodes)} nodes")

        # Step 2.5: Merge tiny nodes
        print("\nStep 2.5: Merging tiny nodes...")
        section_nodes = self.merge_tiny_nodes(section_nodes, min_chars=200)
        print(f"  After merging: {len(section_nodes)} nodes")

        # Step 3: Chunk each section at multiple granularity levels
        print("\nStep 3: Chunking sections at multiple granularity levels...")

        section_chunks = self.chunk_sections(section_nodes)
        print(f"  Created {len(section_chunks)} chunks")
        
        # Step 4: Add metadata to chunks (claim_id, source file, etc.)
        print("\nStep 4: Adding metadata to chunks...")
        for chunk in section_chunks:
            if not hasattr(chunk, 'metadata') or chunk.metadata is None:
                chunk.metadata = {}
            chunk.metadata['claim_id'] = claim_id
            chunk.metadata['source_file'] = str(pdf_path.name)
            chunk.metadata['source_path'] = str(pdf_path)
        
        # Step 5: Embed chunks and save to ChromaDB
        print("\nStep 5: Embedding chunks and saving to ChromaDB...")
        try:
            node_ids = self.chroma_manager.add_nodes(section_chunks)
            print(f"  Successfully embedded and saved {len(node_ids)} chunks to ChromaDB")
            print(f"  Collection: {self.vector_store_config['collection_chunks']}")
            print(f"  Persist directory: {self.vector_store_config['persist_directory']}")
        except Exception as e:
            print(f"  Error saving chunks to ChromaDB: {e}")
            raise

        return {
            'sections': section_nodes,
            'chunks': section_chunks,
            'claim_id': claim_id,
            'node_ids': node_ids
        }


def main():
    """Main entry point for the ingestion pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="PDF Data Ingestion Pipeline using MarkdownNodeParser")
    parser.add_argument(
        "--pdf",
        type=str,
        required=True,
        help="Path to PDF file to process"
    )
    parser.add_argument(
        "--claim-id",
        type=str,
        help="Claim ID (optional, will use filename if not provided)"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = IngestionPipeline()
    
    # Process PDF
    pipeline.process_pdf(args.pdf, claim_id=args.claim_id)

    print("\n" + "="*50)
    print("Processing Complete!")
    print("="*50)


if __name__ == "__main__":
    main()
