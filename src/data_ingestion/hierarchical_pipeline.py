"""Hierarchical chunking pipeline using HierarchicalNodeParser."""

import sys
from pathlib import Path
from typing import Optional, List, Dict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config_loader import load_config, get_chunking_config, get_vector_store_config, get_llm_config
from src.utils.vector_store import ChromaDBManager
from llama_index.core.node_parser import (
    MarkdownNodeParser,
    HierarchicalNodeParser,
    get_leaf_nodes
)
from llama_index.core import Document, Settings
from llama_index.core.schema import BaseNode
from llama_index.llms.openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()


class HierarchicalPipeline:
    """Pipeline for hierarchical chunking using HierarchicalNodeParser."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize hierarchical pipeline."""
        self.parser = MarkdownNodeParser()
        
        # Load configuration
        self.config = load_config(config_path)
        self.chunking_config = get_chunking_config(self.config)
        self.vector_store_config = get_vector_store_config(self.config)
        self.llm_config = get_llm_config(self.config)
        
        # Initialize LLM in Settings for use across the pipeline
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        Settings.llm = OpenAI(
            model=self.llm_config['model'],
            temperature=self.llm_config['temperature'],
            api_key=openai_api_key
        )
        
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
        chunks = []
        for section in sections:
            chunks.extend(self.chunk_section_multi_granularity(section))
        return chunks

    def chunk_section_multi_granularity(self, node: BaseNode) -> List[BaseNode]:
        """
        Chunk a single section/node at multiple granularity levels using HierarchicalNodeParser.
        
        Returns a list of hierarchical chunks.
        """
        section_text = node.get_content()
        
        section_metadata = node.metadata.copy()
        
        # Create Document from section text
        section_doc = Document(text=section_text, metadata=section_metadata)
        
        # Use HierarchicalNodeParser to create hierarchical chunks
        hierarchical_nodes = self.hierarchical_parser.get_nodes_from_documents([section_doc])

        return hierarchical_nodes

    def execute(
        self,
        text: str,
        claim_id: str,
        source_file: str,
        source_path: str,
        split_into_sections: bool = False
    ):
        """
        Execute hierarchical chunking pipeline on text.
        
        Args:
            text: Text to process
            claim_id: Claim ID for metadata
            source_file: Source filename for metadata
            source_path: Source file path for metadata
            split_into_sections: If True, parse text into sections first. If False, use raw text directly.
        
        Returns:
            Dictionary with 'sections', 'chunks', 'claim_id', and 'node_ids'
        """
        print(f"Processing text")
        print(f"Claim ID: {claim_id}")
        print(f"Source file: {source_file}")
        print(f"Split into sections: {split_into_sections}")
        print("-" * 50)

        if split_into_sections:
            # Step 2: Create Document and parse into nodes
            print("\nStep 2: Parsing text into nodes...")
            doc = Document(text=text)
            section_nodes = self.parser.get_nodes_from_documents([doc])
            print(f"  Created {len(section_nodes)} nodes")

            # Step 2.5: Merge tiny nodes
            print("\nStep 2.5: Merging tiny nodes...")
            section_nodes = self.merge_tiny_nodes(section_nodes, min_chars=200)
            print(f"  After merging: {len(section_nodes)} nodes")

            # Step 3: Chunk each section at multiple granularity levels
            print("\nStep 3: Chunking sections at multiple granularity levels...")
            chunks = self.chunk_sections(section_nodes)
            print(f"  Created {len(chunks)} chunks")
        else:
            # Skip section parsing, use raw text directly with hierarchical parser
            print("\nStep 2: Using raw text directly (skipping section parsing)...")
            doc = Document(text=text, metadata={'claim_id': claim_id, 'source_file': source_file, 'source_path': source_path})
            chunks = self.hierarchical_parser.get_nodes_from_documents([doc])
            print(f"  Created {len(chunks)} chunks")
            section_nodes = []  # No section nodes when skipping section parsing
        
        # Step 4: Add metadata to chunks (claim_id, source file, etc.)
        print("\nStep 4: Adding metadata to chunks...")
        for chunk in chunks:
            if not hasattr(chunk, 'metadata') or chunk.metadata is None:
                chunk.metadata = {}
            chunk.metadata['claim_id'] = claim_id
            chunk.metadata['source_file'] = source_file
            chunk.metadata['source_path'] = source_path
        
        # Step 5: Embed chunks and save to ChromaDB
        print("\nStep 5: Embedding chunks and saving to ChromaDB...")
        try:
            node_ids = self.chroma_manager.add_nodes(chunks)
            print(f"  Successfully embedded and saved {len(node_ids)} chunks to ChromaDB as leaf nodes out of {len(chunks)} total nodes")
            print(f"  Collection: {self.vector_store_config['collection_chunks']}")
            print(f"  Persist directory: {self.vector_store_config['persist_directory']}")
        except Exception as e:
            print(f"  Error saving chunks to ChromaDB: {e}")
            raise
        
        return {
            'sections': section_nodes,
            'chunks': chunks,
            'claim_id': claim_id,
            'node_ids': node_ids
        }

    def print_existing_chunks(self):
        """Retrieve and print all existing chunks from ChromaDB for debugging."""
        print("=" * 60)
        print("Retrieving existing chunks from ChromaDB...")
        print("=" * 60)
        
        try:
            # Access the docstore from the index
            docstore = self.chroma_manager.index.storage_context.docstore
            
            if not hasattr(docstore, 'docs') or not docstore.docs:
                print("No chunks found in ChromaDB.")
                return
            
            # Get all nodes from docstore
            all_node_ids = list(docstore.docs.keys())
            print(f"Found {len(all_node_ids)} chunks in ChromaDB\n")
            
            # Get all nodes and determine which are leaf nodes
            all_nodes = [docstore.get_document(node_id) for node_id in all_node_ids]
            leaf_nodes = get_leaf_nodes(all_nodes)
            leaf_node_ids = {node.node_id for node in leaf_nodes}
            
            print("=" * 60)
            print("ALL CHUNKS:")
            print("=" * 60)
            
            for i, node_id in enumerate(all_node_ids, 1):
                try:
                    node = docstore.get_document(node_id)
                    content = node.get_content()
                    content_length = len(content)
                    is_leaf = node_id in leaf_node_ids
                    print(f"\n--- Chunk {i} (ID: {node_id}) ---")
                    print(f"Content: {content}")
                    print(f"Length: {content_length} characters")
                    print(f"Leaf node: {is_leaf}")
                    if hasattr(node, 'metadata') and node.metadata:
                        print(f"Metadata: {node.metadata}")
                    print("-" * 60)
                except Exception as e:
                    print(f"\n--- Chunk {i} (ID: {node_id}) ---")
                    print(f"Error retrieving chunk: {e}")
                    print("-" * 60)
                    
        except Exception as e:
            print(f"Error retrieving chunks from ChromaDB: {e}")
            import traceback
            traceback.print_exc()
