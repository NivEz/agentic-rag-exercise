"""Query Router Agent for determining retrieval strategy."""

import sys
from pathlib import Path
from typing import Literal, Optional
from enum import Enum

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

from src.utils.config_loader import load_config
from src.retrieval.auto_merging_retriever import AutoMergingRetriever


class RetrievalStrategy(str, Enum):
    """Enumeration of retrieval strategies."""
    CHUNKS = "chunks"
    SUMMARIES = "summaries"


# Initialize auto-merging retriever instance (lazy initialization)
_auto_merging_retriever = None


def _get_auto_merging_retriever():
    """Get or create the auto-merging retriever instance."""
    global _auto_merging_retriever
    if _auto_merging_retriever is None:
        _auto_merging_retriever = AutoMergingRetriever()
    return _auto_merging_retriever


@tool
def route_to_chunks(query: str) -> str:
    """
    Route the query to the chunks vector store using auto-merging retrieval.
    Use this when the query requires:
    - Specific details, facts, or precise information
    - Needle-in-haystack type queries
    - Detailed document analysis
    - Specific dates, amounts, or exact information
    
    Args:
        query: The user's query string
        
    Returns:
        The answer from the auto-merging retriever query engine
    """
    print("Query routed to chunks retrieval (auto-merging retrieval)")
    print("-" * 60)
    try:
        # Get the auto-merging retriever and its query engine
        retriever = _get_auto_merging_retriever()
        query_engine = retriever.get_query_engine()
        
        # Query the engine and return the response
        response = query_engine.query(query)
        return str(response)
    except Exception as e:
        return f"Error querying auto-merging retriever: {str(e)}"


@tool
def route_to_summaries(query: str) -> str:
    """
    Route the query to the summaries vector store.
    Use this when the query requires:
    - High-level overview or summary information
    - General understanding of documents
    - Broad questions about document content
    - Timeline or overview questions
    
    Args:
        query: The user's query string
        
    Returns:
        A message indicating summaries are not yet supported
    """
    print("Query routed to summaries retrieval")
    print("-" * 60)
    return "Summaries not supported yet"


class QueryRouterAgent:
    """
    Query Router Agent that determines whether to retrieve context from
    chunks vector store (auto-merging retrieval) or summaries vector store.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize Query Router Agent.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path)
        llm_config = self.config.get('llm', {})
        
        # Initialize LLM for the agent
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Initialize LLM with tools bound to it for agentic behavior
        llm = ChatOpenAI(
            model=llm_config.get('model', 'gpt-4o-mini'),
            temperature=llm_config.get('temperature', 0.0),
            api_key=api_key
        )
        
        # Define tools for routing
        self.tools = [route_to_chunks, route_to_summaries]
        
        # Store tool mapping for execution
        self.tool_map = {
            'route_to_chunks': route_to_chunks,
            'route_to_summaries': route_to_summaries
        }
        
        # Bind tools to the LLM for function calling
        self.llm_with_tools = llm.bind_tools(self.tools)
        
        # System prompt for routing decisions
        self.system_prompt = """You are a query routing agent for an insurance claim document retrieval system.

Your task is to analyze user queries and determine the best retrieval strategy:

1. **Chunks (Auto-Merging Retrieval)**: Use for queries that need:
   - Specific details, facts, or precise information
   - Needle-in-haystack type queries
   - Detailed document analysis
   - Specific dates, amounts, names, or exact information
   - Questions requiring granular document chunks

2. **Summaries**: Use for queries that need:
   - High-level overview or summary information
   - General understanding of documents
   - Broad questions about document content
   - Timeline or overview questions
   - Questions that can be answered from document summaries

Analyze the user's query and use the appropriate routing tool:
- Use `route_to_chunks` if the query requires detailed, specific information
- Use `route_to_summaries` if the query requires high-level overview information

Always use exactly one tool to route the query."""
    
    def route(self, query: str) -> RetrievalStrategy:
        """
        Route a query to determine the appropriate retrieval strategy.
        
        Args:
            query: The user's query string
            
        Returns:
            RetrievalStrategy enum value indicating chunks or summaries
        """
        # Create messages for the LLM
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=query)
        ]
        
        # Invoke the LLM with tools
        response = self.llm_with_tools.invoke(messages)
        
        # Check if the LLM called a tool
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_call = response.tool_calls[0]
            # Handle both dict and object formats
            if isinstance(tool_call, dict):
                tool_name = tool_call.get('name', '')
                tool_args = tool_call.get('args', {})
            else:
                tool_name = getattr(tool_call, 'name', '')
                tool_args = getattr(tool_call, 'args', {})
            
            # Execute the tool using the tool map
            if tool_name in self.tool_map:
                tool = self.tool_map[tool_name]
                # Use tool_args if available, otherwise use query
                if tool_args and 'query' in tool_args:
                    output = tool.invoke(tool_args)
                else:
                    output = tool.invoke({"query": query})
                print(f"\nLLM response: {output}")
                
                # Return appropriate strategy
                if tool_name == 'route_to_chunks':
                    return RetrievalStrategy.CHUNKS
                elif tool_name == 'route_to_summaries':
                    return RetrievalStrategy.SUMMARIES
        
        # If no tool was called, ask user to refine the question
        print("\nUnable to determine routing strategy. Please refine your question to be more specific.")
