"""Utility functions for LLM interactions."""

import os
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()


def get_llm(model: str, temperature: float = 0.0) -> OpenAI:
    """Get OpenAI LLM instance."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    return OpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key
    )


def get_embedding_model(model: str = "text-embedding-3-small") -> OpenAIEmbedding:
    """Get OpenAI embedding model instance."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    return OpenAIEmbedding(
        model=model,
        api_key=api_key
    )
