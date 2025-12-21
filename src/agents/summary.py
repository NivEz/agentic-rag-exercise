"""Summary Agent for retrieving context from summaries and answering broad questions."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent

from src.utils.config_loader import load_config
from src.retrieval.summaries_retriever import SummariesRetriever
from src.agents.response_collector import ResponseCollector


class SummaryAgent:
    """
    LangChain agent that uses summaries retrieval to answer user queries.
    The agent has two tools:
    - retrieve_context: Retrieves document-level summaries (high-level overviews)
    - retrieve_detailed_context: Retrieves chunk-level summaries (detailed sections)
    
    The agent defaults to using document-level summaries for broad questions,
    but can fall back to chunk-level summaries when more detail is needed.
    """
    
    def __init__(self, config_path: str = "config/config.yaml", collector: ResponseCollector = None):
        """
        Initialize Summary Agent.
        
        Args:
            config_path: Path to configuration file
            collector: Optional ResponseCollector instance for tracking responses
        """
        # Load configuration
        self.config = load_config(config_path)
        llm_config = self.config.get('llm', {})
        
        # Store collector
        self.collector = collector
        
        # Initialize summaries retriever
        self.retriever = SummariesRetriever(config_path=config_path)
        
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
        
        # Get default top_k from config
        retrieval_config = self.config.get('retrieval', {})
        default_top_k = retrieval_config.get('default_top_k', 5)
        
        # Create the retrieve_context tool using closure to access self.retriever and self.collector
        # This tool retrieves document-level summaries (high-level overviews)
        @tool
        def retrieve_context(query: str) -> str:
            """
            Retrieve document-level summaries (high-level overviews of entire documents).
            Use ONLY for very broad questions with NO specific details mentioned.
            """
            try:
                # Use query_by_summary_level to get document-level summaries
                results = self.retriever.query_by_summary_level(
                    query=query,
                    summary_level='document',
                    top_k=default_top_k,
                    return_text=True
                )
                
                # Combine all retrieved summaries into a single string
                context_parts = [result['text'] for result in results]
                
                # Collect contexts if collector is available
                if self.collector:
                    self.collector.collect_contexts(context_parts)
                
                context = "\n\n".join(context_parts)
                return context if context else "No relevant document summaries found."
            except Exception as e:
                return f"Error retrieving document summaries: {str(e)}"
        
        # Create a tool for retrieving chunk-level summaries (more detailed)
        @tool
        def retrieve_detailed_context(query: str) -> str:
            """
            Retrieve chunk-level summaries (detailed summaries of specific document sections).
            Use for questions mentioning specific details, names, dates, amounts, locations, or entities.
            This is the DEFAULT tool for most questions.
            """
            try:
                # Use query_by_summary_level to get chunk-level summaries
                results = self.retriever.query_by_summary_level(
                    query=query,
                    summary_level='chunk',
                    top_k=default_top_k,
                    return_text=True
                )
                
                # Combine all retrieved summaries into a single string
                context_parts = [result['text'] for result in results]
                
                # Collect contexts if collector is available
                if self.collector:
                    self.collector.collect_contexts(context_parts)
                
                context = "\n\n".join(context_parts)
                return context if context else "No relevant chunk summaries found."
            except Exception as e:
                return f"Error retrieving chunk summaries: {str(e)}"
        
        # Define the tools
        self.tools = [retrieve_context, retrieve_detailed_context]
        
        # System prompt for the agent
        system_prompt = """You are a helpful assistant that answers questions based on retrieved document summaries.

You have access to two retrieval tools. CRITICAL: Choose the correct tool based on the question type.

TOOL SELECTION RULES:

1. retrieve_context (document-level summaries):
   ONLY use for questions that ask for:
   - "What is this document about?" or "Give me an overview"
   - "What happened overall?" (very broad, no specifics)
   - "What is the general timeline?" (high-level only)
   - Questions with NO specific nouns, names, dates, or details mentioned
   
   Examples of when to use retrieve_context:
   - "What is this document about?"
   - "Give me a summary"
   - "What happened?"
   - "What is the overall timeline?"

2. retrieve_detailed_context (chunk-level summaries):
   USE THIS for questions that mention:
   - Specific people, organizations, or entities (names)
   - Specific dates, times, or periods
   - Specific events, incidents, or occurrences
   - Specific amounts, values, numbers, or quantities
   - Specific locations, places, or addresses
   - Specific topics, subjects, or themes
   - Any question with a specific noun, proper noun, or detail
   
   Examples of when to use retrieve_detailed_context:
   - "What happened with the accident on [date]?"
   - "Tell me about [person's name]"
   - "What was the claim amount?"
   - "Where did the incident occur?"
   - "What happened during [specific event]?"
   - "Tell me about [specific topic]"
   - "What are the details about [something specific]?"

DECISION PROCESS:
1. Analyze the user's question carefully
2. If the question mentions ANY specific noun, name, date, amount, location, or detail → use retrieve_detailed_context
3. If the question is ONLY asking for a very broad, general overview with NO specifics → use retrieve_context
4. When in doubt, use retrieve_detailed_context (it's better to get specific details than miss them)

After retrieving context:
- Analyze the retrieved summaries carefully
- Answer the user's question based on the summary context provided
- If the summaries don't contain enough information, say so clearly
- You can use both tools if needed, but start with the appropriate one based on the question type"""
        
        # Create the agent
        self.agent = create_agent(
            model=llm,
            tools=self.tools,
            system_prompt=system_prompt
        )
    
    def answer(self, query: str) -> str:
        """
        Answer a user query using the agent.
        
        Args:
            query: The user's question
            
        Returns:
            The agent's answer to the query
        """
        # Reset collector if available
        if self.collector:
            self.collector.reset()
        
        result = ""
        # Invoke the agent with the user query
        for event in self.agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
        ):
            result = event["messages"][-1]
            result.pretty_print()
            if hasattr(result, 'content') and result.content:
                if not hasattr(result, 'tool_calls') or not result.tool_calls:
                    result_content = str(result.content)
                    result = result_content
                    # Collect response if collector is available
                    if self.collector:
                        self.collector.collect_response(result_content)
        
        return result
    
    def get_agent(self):
        """Get the underlying agent."""
        return self.agent


def main():
    """Main function to interact with the SummaryAgent using command line input."""
    print("Initializing SummaryAgent...")
    agent = SummaryAgent()
    print("SummaryAgent ready! Type your questions (or 'quit'/'exit' to stop).\n")
    
    while True:
        try:
            query = input("Query: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            print("\nProcessing...")
            agent.answer(query)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}\n")


if __name__ == "__main__":
    main()

