"""Query Router Agent for determining retrieval strategy."""

import sys
from pathlib import Path
from enum import Enum

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent

from src.utils.config_loader import load_config
from src.agents.needle import NeedleAgent
from src.agents.summary import SummaryAgent


class RetrievalStrategy(str, Enum):
    """Enumeration of retrieval strategies."""
    CHUNKS = "chunks"
    SUMMARIES = "summaries"


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
        
        # Initialize NeedleAgent instance
        self.needle_agent = NeedleAgent(config_path=config_path)
        
        # Initialize SummaryAgent instance
        self.summary_agent = SummaryAgent(config_path=config_path)
        
        # Initialize LLM for the agent
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Initialize LLM
        llm = ChatOpenAI(
            model=llm_config.get('model', 'gpt-4o-mini'),
            temperature=llm_config.get('temperature', 0.0),
            api_key=api_key
        )
        
        # Create the route_to_needle tool using closure to access self.needle_agent
        @tool
        def route_to_needle(query: str) -> str:
            """
            Route the query to the needle agent using auto-merging retrieval.
            Use this when the query requires:
            - Specific details, facts, or precise information
            - Needle-in-haystack type queries
            - Detailed document analysis
            - Specific dates, amounts, or exact information
            
            Args:
                query: The user's query string
                
            Returns:
                The answer from the needle agent
            """
            print("Query routed to needle agent (auto-merging retrieval)")
            print("-" * 60)
            try:
                response = self.needle_agent.answer(query)
                return response
            except Exception as e:
                return f"Error querying needle agent: {str(e)}"
        
        # Create the route_to_summaries tool using closure to access self.summary_agent
        @tool
        def route_to_summaries(query: str) -> str:
            """
            Route the query to the summary agent using summaries retrieval.
            Use this when the query requires:
            - High-level overview or summary information
            - General understanding of documents
            - Broad questions about document content
            - Timeline or overview questions
            
            Args:
                query: The user's query string
                
            Returns:
                The answer from the summary agent
            """
            print("Query routed to summary agent (summaries retrieval)")
            print("-" * 60)
            try:
                response = self.summary_agent.answer(query)
                return response
            except Exception as e:
                return f"Error querying summary agent: {str(e)}"
        
        # Define tools for routing
        self.tools = [route_to_needle, route_to_summaries]
        
        # System prompt for routing decisions
        system_prompt = """You are a query routing agent for an insurance claim document retrieval system.

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
- Use `route_to_needle` if the query requires detailed, specific information
- Use `route_to_summaries` if the query requires high-level overview information
- If you are not sure about the routing strategy, simply say "Query is not clear, please provide more information"


Always use exactly one tool to route the query."""
        
        # Create the agent
        self.agent = create_agent(
            model=llm,
            tools=self.tools,
            system_prompt=system_prompt
        )
    
    def route(self, query: str) -> RetrievalStrategy:
        """
        Route a query to determine the appropriate retrieval strategy.
        
        Args:
            query: The user's query string
            
        Returns:
            RetrievalStrategy enum value indicating chunks or summaries
        """
        # Invoke the agent with the user query
        self.agent.invoke({"messages": [{"role": "user", "content": query}]})
