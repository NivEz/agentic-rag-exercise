"""Vector store utilities for ChromaDB integration."""

import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import BaseNode, TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
import chromadb
from chromadb.config import Settings

load_dotenv()


def get_embedding_model(model: str = "text-embedding-3-small") -> OpenAIEmbedding:
    """Get OpenAI embedding model instance."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    return OpenAIEmbedding(
        model=model,
        api_key=api_key
    )


def get_chroma_client(persist_directory: str) -> chromadb.ClientAPI:
    """Initialize and return ChromaDB client."""
    persist_path = Path(persist_directory)
    persist_path.mkdir(parents=True, exist_ok=True)
    
    chroma_client = chromadb.PersistentClient(
        path=str(persist_path),
        settings=Settings(anonymized_telemetry=False)
    )
    
    return chroma_client


def get_or_create_collection(
    chroma_client: chromadb.ClientAPI,
    collection_name: str
) -> chromadb.Collection:
    """Get existing collection or create a new one."""
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception:
        collection = chroma_client.create_collection(name=collection_name)
    
    return collection


def get_vector_store_index(
    collection: chromadb.Collection,
    embedding_model: OpenAIEmbedding
) -> VectorStoreIndex:
    """Create or get VectorStoreIndex from ChromaDB collection."""
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create index with embedding model
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embedding_model
    )
    
    return index


class ChromaDBManager:
    """Manager class for ChromaDB operations."""
    
    def __init__(
        self,
        persist_directory: str,
        collection_name: str,
        embedding_model_name: str = "text-embedding-3-small"
    ):
        """Initialize ChromaDB manager."""
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = get_embedding_model(embedding_model_name)
        
        # Initialize ChromaDB client and collection
        self.chroma_client = get_chroma_client(persist_directory)
        self.collection = get_or_create_collection(
            self.chroma_client,
            collection_name
        )
        
        # Initialize vector store index
        self.index = get_vector_store_index(
            self.collection,
            self.embedding_model
        )
    
    def add_nodes(self, nodes: List[BaseNode]) -> List[str]:
        """
        Add nodes to ChromaDB and return their IDs.
        Nodes will be automatically embedded using the configured embedding model.
        
        Args:
            nodes: List of nodes to add
            
        Returns:
            List of node IDs
        """
        if not nodes:
            return []
        
        # Insert nodes into the index (embedding happens automatically)
        self.index.insert_nodes(nodes, show_progress=True)
        
        # Return node IDs
        return [node.node_id for node in nodes]
    
    def add_documents(self, documents: List[BaseNode]) -> List[str]:
        """
        Add documents (nodes) to ChromaDB.
        Alias for add_nodes for consistency.
        """
        return self.add_nodes(documents)
    
    def get_index(self) -> VectorStoreIndex:
        """Get the vector store index."""
        return self.index

