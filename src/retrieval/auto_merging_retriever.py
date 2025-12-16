"""Auto-Merging Retriever for hierarchical chunk retrieval."""

import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llama_index.core.retrievers import AutoMergingRetriever as LlamaAutoMergingRetriever
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore, BaseNode
from llama_index.core import VectorStoreIndex

from src.utils.config_loader import load_config, get_chunking_config, get_vector_store_config
from src.utils.vector_store import ChromaDBManager


class AutoMergingRetriever:
    """
    Auto-Merging Retriever that retrieves leaf nodes and merges them
    back into parent nodes when they share a common parent.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize Auto-Merging Retriever.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path)
        self.chunking_config = get_chunking_config(self.config)
        self.vector_store_config = get_vector_store_config(self.config)
        
        # Initialize ChromaDB manager to get the index
        self.chroma_manager = ChromaDBManager(
            persist_directory=self.vector_store_config['persist_directory'],
            collection_name=self.vector_store_config['collection_chunks'],
            embedding_model_name=self.vector_store_config['embedding_model']
        )
        
        # Get the vector store index
        self.index = self.chroma_manager.get_index()
        
        # Initialize HierarchicalNodeParser (same as in pipeline)
        # This is needed to understand the hierarchy for merging
        chunk_sizes = [
            self.chunking_config['large_chunk_size'],
            self.chunking_config['medium_chunk_size'],
            self.chunking_config['small_chunk_size']
        ]
        
        self.hierarchical_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=chunk_sizes,
            chunk_overlap=self.chunking_config['chunk_overlap']
        )
        
        # Create base vector retriever (retrieves leaf nodes)
        # Retrieve more initially to allow for merging
        self.base_retriever = self.index.as_retriever(similarity_top_k=20)
        
        # Create Auto-Merging Retriever
        # It will retrieve leaf nodes and merge them into parent nodes when appropriate
        # Note: Requires parent nodes to be stored in docstore (from hierarchical ingestion)
        try:
            self.auto_merging_retriever = LlamaAutoMergingRetriever(
                self.base_retriever,
                self.index.storage_context,
                verbose=True,
                # Optional: similarity threshold for merging (default is usually fine)
                # similarity_cutoff=0.7
            )
        except Exception as e:
            print(f"Warning: Auto-Merging Retriever initialization issue: {e}")
            print("This may occur if parent nodes are not in the docstore.")
            print("Please re-run the ingestion pipeline to ensure all hierarchical nodes are stored.")
            raise
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[NodeWithScore]:
        """
        Retrieve nodes for a given query using Auto-Merging Retriever.
        
        The retriever will:
        1. Retrieve leaf nodes (small chunks) from the vector store
        2. Check if multiple leaf nodes share a common parent
        3. Merge them back into parent nodes (medium/large chunks) when appropriate
        
        Args:
            query: Free text query string
            top_k: Number of results to return (default: 5)
            
        Returns:
            List of NodeWithScore objects (may be merged parent nodes)
        """
        # Set similarity_top_k for the base retriever
        # We retrieve more leaf nodes initially to allow for merging opportunities
        # The Auto-Merging Retriever will then merge related leaf nodes into parents
        self.base_retriever.similarity_top_k = max(top_k * 4, 20)  # Retrieve more to allow merging
        
        # Retrieve nodes (Auto-Merging Retriever will merge leaf nodes into parents)
        nodes = self.auto_merging_retriever.retrieve(query)
        
        # Limit to top_k results
        return nodes[:top_k]
    
    def query(
        self,
        query: str,
        top_k: int = 5,
        return_text: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Query the retriever and return formatted results.
        
        Args:
            query: Free text query string
            top_k: Number of results to return (default: 5)
            return_text: If True, return text content; if False, return full node objects
            
        Returns:
            List of dictionaries containing retrieved information
        """
        nodes = self.retrieve(query, top_k=top_k)
        
        results = []
        for node_with_score in nodes:
            node = node_with_score.node
            
            result = {
                'score': node_with_score.score,
                'text': node.get_content(),
                'metadata': node.metadata if hasattr(node, 'metadata') else {},
                'node_id': node.node_id if hasattr(node, 'node_id') else None
            }
            
            if not return_text:
                result['node'] = node
            
            results.append(result)
        
        return results
    
    def get_retriever(self) -> LlamaAutoMergingRetriever:
        """Get the underlying Auto-Merging Retriever instance."""
        return self.auto_merging_retriever
    
    def get_query_engine(self) -> RetrieverQueryEngine:
        """
        Get a query engine that can be used for more advanced queries.
        Requires LLM to be configured.
        """
        from src.utils.llm_utils import get_llm
        
        llm_config = self.config.get('llm', {})
        llm = get_llm(
            model=llm_config.get('model', 'gpt-4o-mini'),
            temperature=llm_config.get('temperature', 0.0)
        )
        
        query_engine = RetrieverQueryEngine.from_args(
            retriever=self.auto_merging_retriever,
            llm=llm
        )
        
        return query_engine

