"""Summaries Retriever for retrieving document summaries."""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator

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
    
    def _create_metadata_filters(self, metadata_filters: Dict[str, Any]) -> MetadataFilters:
        """
        Convert dictionary of metadata filters to LlamaIndex MetadataFilters.
        
        Args:
            metadata_filters: Dictionary of metadata key-value pairs to filter by
            
        Returns:
            MetadataFilters object for use with retriever
        """
        filters = []
        for key, value in metadata_filters.items():
            filters.append(
                MetadataFilter(
                    key=key,
                    value=value,
                    operator=FilterOperator.EQ
                )
            )
        return MetadataFilters(filters=filters)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[NodeWithScore]:
        """
        Retrieve summary nodes for a given query using vector similarity search.
        
        Uses LlamaIndex's built-in metadata filtering when filters are provided,
        which filters at the vector store level (ChromaDB) for better performance.
        
        Args:
            query: Free text query string
            top_k: Number of results to return (default: 5)
            metadata_filters: Optional dictionary of metadata filters to apply.
                             Examples:
                             - {'claim_id': 'claim_123'}  # Filter by claim_id
                             - {'summary_level': 'document'}  # Filter by summary level
                             - {'source_file': 'report.pdf'}  # Filter by source file
                             Multiple filters are combined with AND logic.
                             
        Returns:
            List of NodeWithScore objects containing summaries that match the filters
        """
        # Create retriever with metadata filters if provided
        if metadata_filters:
            # Convert dictionary filters to LlamaIndex MetadataFilters
            filters = self._create_metadata_filters(metadata_filters)
            # Create a new retriever with filters applied
            retriever = self.index.as_retriever(
                similarity_top_k=top_k,
                filters=filters
            )
        else:
            # Use base retriever without filters
            retriever = self.base_retriever
            retriever.similarity_top_k = top_k
        
        # Retrieve nodes (filtering happens at vector store level)
        nodes = retriever.retrieve(query)
        
        return nodes
    
    def query(
        self,
        query: str,
        top_k: int = 5,
        return_text: bool = True,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query the retriever and return formatted results.
        
        Args:
            query: Free text query string
            top_k: Number of results to return (default: 5)
            return_text: If True, return text content; if False, return full node objects
            metadata_filters: Optional dictionary of metadata filters to apply.
                             Examples:
                             - {'claim_id': 'claim_123'}  # Filter by claim_id
                             - {'summary_level': 'document'}  # Filter by summary level
                             - {'source_file': 'report.pdf'}  # Filter by source file
                             Multiple filters are combined with AND logic.
            
        Returns:
            List of dictionaries containing retrieved summary information
        """
        nodes = self.retrieve(query, top_k=top_k, metadata_filters=metadata_filters)
        
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
    
    def query_by_claim_id(
        self,
        query: str,
        claim_id: str,
        top_k: int = 5,
        return_text: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Convenience method to query summaries for a specific claim ID.
        
        Args:
            query: Free text query string
            claim_id: Claim ID to filter by
            top_k: Number of results to return (default: 5)
            return_text: If True, return text content; if False, return full node objects
            
        Returns:
            List of dictionaries containing retrieved summary information for the claim
        """
        return self.query(
            query=query,
            top_k=top_k,
            return_text=return_text,
            metadata_filters={'claim_id': claim_id}
        )
    
    def query_by_summary_level(
        self,
        query: str,
        summary_level: str,
        top_k: int = 5,
        return_text: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Convenience method to query summaries by summary level (chunk or document).
        
        Args:
            query: Free text query string
            summary_level: Summary level to filter by ('chunk' or 'document')
            top_k: Number of results to return (default: 5)
            return_text: If True, return text content; if False, return full node objects
            
        Returns:
            List of dictionaries containing retrieved summary information
        """
        return self.query(
            query=query,
            top_k=top_k,
            return_text=return_text,
            metadata_filters={'summary_level': summary_level}
        )
    
    def get_retriever(self) -> VectorIndexRetriever:
        """Get the underlying Vector Index Retriever instance."""
        return self.base_retriever
    
    def get_index(self) -> VectorStoreIndex:
        """Get the underlying Vector Store Index."""
        return self.index

