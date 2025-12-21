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
from src.agents.response_collector import ResponseCollector


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
        
        # Create a shared collector for sub-agents (questions are asked one by one)
        self.collector = ResponseCollector()
        
        # Initialize NeedleAgent instance with shared collector
        self.needle_agent = NeedleAgent(config_path=config_path, collector=self.collector)
        
        # Initialize SummaryAgent instance with shared collector
        self.summary_agent = SummaryAgent(config_path=config_path, collector=self.collector)
        
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

Your task is to analyze user queries and route them to the most appropriate retrieval strategy.

## Routing Decision Framework

### Route to `route_to_needle` (Auto-Merging Retrieval) when:
- Query seeks **specific facts, numbers, or exact details** (e.g., claim amounts, policy numbers, dates, names)
- Query requires **precise information extraction** from document sections
- Query is a **"needle-in-haystack"** type (finding specific information within large documents)
- Query asks about **granular details** that require searching individual document chunks
- Query needs **exact quotes or verbatim text** from documents

**Examples:**
- "What was the claim amount for policy #12345?"
- "When did the incident occur?"
- "What is the exact wording in section 3.2?"

### Route to `route_to_summaries` when:
- Query seeks **high-level understanding** or document overview
- Query asks **"What is the document about?"** or similar general content questions
- Query asks about **general themes, patterns, or trends** across documents
- Query requires **contextual understanding** rather than specific facts
- Query asks for **summaries, timelines, or broad analysis**
- Query can be answered from **document-level summaries** without diving into chunks
- Query uses phrases like "tell me about", "what is about", "overview", "summary", "describe"

**Examples:**
- "What is the document about?"
- "What are the main types of claims in these documents?"
- "Give me an overview of all the insurance claims"
- "What is the general timeline of events?"
- "Tell me about the documents"
- "What topics are covered?"

## Decision Rules

1. **Always use exactly one tool** - either `route_to_needle` or `route_to_summaries`
2. **General document questions default to summaries**: Questions asking "what is about", "tell me about", or seeking general understanding should route to `route_to_summaries`
3. **When uncertain between strategies**: Default to `route_to_needle` only if the query could require specific details. Default to `route_to_summaries` if the query is general or asks about document content
4. **Only mark as unclear**: If the query is completely ambiguous, nonsensical, or lacks any meaningful content (not just general questions)
5. **Consider query intent**: Focus on what the user is trying to accomplish - general understanding → summaries, specific facts → chunks

**Important**: "What is the document about?" and similar general content questions are clear and should route to `route_to_summaries`.

Analyze the query carefully and route accordingly."""
        
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
        response = self.agent.invoke({"messages": [{"role": "user", "content": query}]}, return_intermediate_steps=True)

        # Check if the agent determined the query is unclear
        # The agent might respond with text instead of using a tool
        if response and "messages" in response:
            # Check if any tool was called
            tool_called = False
            unclear_message = None
            
            for message in response["messages"]:
                # Check if this message contains a tool call
                if hasattr(message, "tool_calls") and message.tool_calls:
                    tool_called = True
                    break
                # Check if this is a text response indicating unclear query
                if hasattr(message, "content") and message.content:
                    content = str(message.content).lower()
                    if "not clear" in content or "provide more information" in content:
                        unclear_message = message.content
                        break
            
            # If no tool was called and we found an unclear message, print it
            if not tool_called and unclear_message:
                print("Query is not clear, please provide more information")

    def answer_with_contexts(self, query: str) -> dict:
        """
        Answer a query via the router and return both the answer and retrieved contexts.
        
        Args:
            query: The user's query string
            
        Returns:
            Dictionary with 'text_response' (str) and 'contexts' (list[str]) keys
        """
        # Reset the shared collector for a new query
        self.collector.reset()
        
        # Invoke the router agent
        response = self.agent.invoke({"messages": [{"role": "user", "content": query}]})
        
        # Get results from the shared collector (whichever sub-agent was used will have populated it)
        result = self.collector.get_result()
        
        # If no contexts were collected, the router might not have called a tool
        if not result['contexts']:
            # Try to extract answer from router response
            router_answer = ""
            if response and "messages" in response:
                for message in reversed(response["messages"]):
                    if hasattr(message, "content") and message.content:
                        if not hasattr(message, "tool_calls") or not message.tool_calls:
                            router_answer = str(message.content)
                            break
            
            if router_answer:
                result['text_response'] = router_answer
        
        return result
