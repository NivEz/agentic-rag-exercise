"""Main data ingestion pipeline using llama_index MarkdownNodeParser."""

import sys
from pathlib import Path
from typing import Optional, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_ingestion.pdf_processor import PDFProcessor
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core import Document
from llama_index.core.schema import TextNode, BaseNode


class IngestionPipeline:
    """Main pipeline for ingesting PDFs using MarkdownNodeParser."""
    
    def __init__(self):
        """Initialize ingestion pipeline."""
        self.pdf_processor = PDFProcessor()
        self.parser = MarkdownNodeParser()

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
        nodes = self.parser.get_nodes_from_documents([doc])
        print(f"  Created {len(nodes)} nodes")

        # Step 2.5: Merge tiny nodes
        print("\nStep 2.5: Merging tiny nodes...")
        nodes = self.merge_tiny_nodes(nodes, min_chars=200)
        print(f"  After merging: {len(nodes)} nodes")

        # Step 3: Print nodes
        print("\nStep 3: Printing nodes...")
        print("=" * 50)
        for i, node in enumerate(nodes, 1):
            print(f"\nNode {i}:")
            print(f"  Node ID: {node.node_id}")
            print(f"  Node Metadata: {node.metadata}")
            content = node.get_content()
            print(f"  Content Preview: {content}")
            print(f"  Content Length: {len(content)} characters")
            print("-" * 50)

        print(f"\n\nTotal nodes parsed: {len(nodes)}")
        return nodes


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
