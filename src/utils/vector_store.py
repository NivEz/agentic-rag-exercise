"""Vector store utilities for ChromaDB integration."""

import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import get_leaf_nodes
from llama_index.core.schema import BaseNode
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
    embedding_model: OpenAIEmbedding,
    persist_directory: str
) -> VectorStoreIndex:
    """
    Create or get VectorStoreIndex from ChromaDB collection.
    Uses LlamaIndex's default JSON persistence for docstore.
    """
    vector_store = ChromaVectorStore(chroma_collection=collection)
    
    # Load existing storage context if it exists, otherwise create new one
    # This automatically handles JSON persistence for docstore
    persist_path = Path(persist_directory)
    if persist_path.exists() and (persist_path / "docstore.json").exists():
        # Load existing storage context (this loads docstore from JSON)
        existing_context = StorageContext.from_defaults(
            persist_dir=persist_directory
        )
        
        # Get the loaded docstore and ensure it's fully initialized
        docstore = existing_context.docstore
        index_store = existing_context.index_store
        graph_store = existing_context.graph_store
        
        # Verify docstore has nodes and test access
        if hasattr(docstore, 'docs'):
            print(f"Docstore has {len(docstore.docs)} documents")
            # Test accessing a few nodes to ensure they're loaded
            test_ids = list(docstore.docs.keys())[:5]
            for doc_id in test_ids:
                try:
                    node = docstore.get_document(doc_id)
                    print(f"Successfully loaded node: {doc_id}")
                except Exception as e:
                    print(f"Error loading node {doc_id}: {e}")
        
        # Create new storage context with our vector store and the loaded docstore
        # Important: Pass persist_dir to ensure docstore persistence is maintained
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            docstore=docstore,
            index_store=index_store,
            graph_store=graph_store,
            persist_dir=persist_directory  # Keep persist_dir for future saves
        )
        
        # Verify the new context has access to the docstore
        if hasattr(storage_context.docstore, 'docs'):
            print(f"New storage context docstore has {len(storage_context.docstore.docs)} documents")
    else:
        # Create new storage context
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=persist_directory  # Set persist_dir for future saves
        )
    
    # Create index with embedding model
    # Use VectorStoreIndex.from_vector_store to create index from ChromaDB vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embedding_model
    )
    
    # CRITICAL FIX: VectorStoreIndex.from_vector_store creates a new StorageContext
    # that doesn't preserve the docstore. We need to restore it manually.
    if hasattr(storage_context, 'docstore') and hasattr(storage_context.docstore, 'docs'):
        original_doc_count = len(storage_context.docstore.docs)
        if original_doc_count > 0:
            # Check if index lost the docstore
            index_doc_count = 0
            if hasattr(index.storage_context, 'docstore') and hasattr(index.storage_context.docstore, 'docs'):
                index_doc_count = len(index.storage_context.docstore.docs)
            
            if index_doc_count == 0:
                # Restore the docstore by replacing it in the index's storage context
                print(f"Restoring docstore with {original_doc_count} documents to index")
                index.storage_context.docstore = storage_context.docstore
                print(f"Docstore restored successfully")
            else:
                print(f"Index created with docstore containing {index_doc_count} documents")
    
    # Final verification
    if hasattr(index.storage_context, 'docstore') and hasattr(index.storage_context.docstore, 'docs'):
        doc_count = len(index.storage_context.docstore.docs)
        print(f"Final verification: Index docstore has {doc_count} documents")
        if doc_count > 0:
            test_id = list(index.storage_context.docstore.docs.keys())[0]
            try:
                test_node = index.storage_context.docstore.get_document(test_id)
                print(f"Successfully verified docstore access with test node: {test_id}")
            except Exception as e:
                print(f"Warning: Could not access test node {test_id} from index docstore: {e}")
    
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
        
        # Initialize vector store index with persistent docstore
        self.index = get_vector_store_index(
            self.collection,
            self.embedding_model,
            self.persist_directory
        )
    
    def add_nodes(self, nodes: List[BaseNode]) -> List[str]:
        """
        Add nodes to ChromaDB and return their IDs.
        Only leaf nodes are stored in the vector store, all nodes are stored in docstore.
        
        Args:
            nodes: List of nodes to add (should include all hierarchical nodes from HierarchicalNodeParser)
            
        Returns:
            List of node IDs
        """
        if not nodes:
            return []
        
        # Get leaf nodes (only these go to vector store)
        leaf_nodes = get_leaf_nodes(nodes)
        
        # Store all nodes in docstore (including parent nodes)
        # This is needed for Auto-Merging Retriever to work
        for node in nodes:
            self.index.storage_context.docstore.add_documents([node], allow_update=True)
        
        # Insert only leaf nodes into vector store (they get embedded)
        if leaf_nodes:
            self.index.insert_nodes(leaf_nodes, show_progress=True)
        
        # Persist docstore (automatically saves as JSON)
        self.index.storage_context.persist(persist_dir=self.persist_directory)
        
        # Return node IDs from original nodes
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

