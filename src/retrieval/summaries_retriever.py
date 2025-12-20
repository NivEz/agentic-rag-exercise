"""Summaries Retriever for retrieving document summaries."""

import sys
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core import VectorStoreIndex

from src.utils.config_loader import load_config, get_vector_store_config
from src.utils.vector_store import ChromaDBManager


class SummariesRetriever:
    """
    Retriever for document summaries using vector similarity search.
    Summaries are stored as flat nodes (no hierarchy), so this is simpler
    than the AutoMergingRetriever.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize Summaries Retriever.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path)
        self.vector_store_config = get_vector_store_config(self.config)
        
        # Initialize ChromaDB manager for summaries collection
        self.chroma_manager = ChromaDBManager(
            persist_directory=self.vector_store_config['persist_directory'],
            collection_name=self.vector_store_config['collection_summaries'],
            embedding_model_name=self.vector_store_config['embedding_model']
        )
        
        # Get the vector store index
        self.index = self.chroma_manager.get_index()
        
        # Create base vector retriever
        self.base_retriever = self.index.as_retriever()
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[NodeWithScore]:
        """
        Retrieve summary nodes for a given query using vector similarity search.
        
        Args:
            query: Free text query string
            top_k: Number of results to return (default: 5)
            
        Returns:
            List of NodeWithScore objects containing summaries
        """
        # Set similarity_top_k for the retriever
        self.base_retriever.similarity_top_k = top_k
        
        # Retrieve nodes
        nodes = self.base_retriever.retrieve(query)
        
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
            List of dictionaries containing retrieved summary information
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
    
    def get_retriever(self) -> VectorIndexRetriever:
        """Get the underlying Vector Index Retriever instance."""
        return self.base_retriever
    
    def get_index(self) -> VectorStoreIndex:
        """Get the underlying Vector Store Index."""
        return self.index

