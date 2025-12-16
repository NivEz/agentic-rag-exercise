"""Configuration loader utility."""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_chunking_config(config: Dict[str, Any]) -> Dict[str, int]:
    """Extract chunking configuration from config dict."""
    chunking_config = config.get('chunking', {})
    return {
        'small_chunk_size': chunking_config.get('small_chunk_size'),
        'medium_chunk_size': chunking_config.get('medium_chunk_size'),
        'large_chunk_size': chunking_config.get('large_chunk_size'),
        'chunk_overlap': chunking_config.get('chunk_overlap')
    }


def get_vector_store_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract vector store configuration from config dict."""
    vector_store_config = config.get('vector_store', {})
    llm_config = config.get('llm', {})
    return {
        'provider': vector_store_config.get('provider'),
        'persist_directory': vector_store_config.get('persist_directory'),
        'collection_chunks': vector_store_config.get('collection_chunks'),
        'collection_summaries': vector_store_config.get('collection_summaries'),
        'embedding_model': llm_config.get('embedding_model')
    }
