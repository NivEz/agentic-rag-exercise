"""Needle Agent for retrieving context and answering queries."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent

from src.utils.config_loader import load_config
from src.retrieval.auto_merging_retriever import AutoMergingRetriever
from src.agents.response_collector import ResponseCollector


class NeedleAgent:
    """
    Simple LangChain agent that uses auto-merging retrieval to answer user queries.
    The agent has a single tool called retrieve_context that retrieves relevant
    document context, then uses that context to answer the user's question.
    """
    
    def __init__(self, config_path: str = "config/config.yaml", collector: ResponseCollector = None):
        """
        Initialize Needle Agent.
        
        Args:
            config_path: Path to configuration file
            collector: Optional ResponseCollector instance for tracking responses
        """
        # Load configuration
        self.config = load_config(config_path)
        llm_config = self.config.get('llm', {})
        
        # Store collector
        self.collector = collector
        
        # Initialize auto-merging retriever instance
        self.retriever = AutoMergingRetriever(config_path=config_path)
        
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
        
        # Create the retrieve_context tool using closure to access self.retriever and self.collector
        @tool
        def retrieve_context(query: str) -> str:
            """Retrieve relevant context from documents using auto-merging retrieval."""
            try:
                # Use the query method which returns formatted results
                results = self.retriever.query(query, top_k=5, return_text=True)
                
                # Combine all retrieved contexts into a single string
                context_parts = [result['text'] for result in results]
                
                # Collect contexts if collector is available
                if self.collector:
                    self.collector.collect_contexts(context_parts)
                
                context = "\n\n".join(context_parts)
                return context if context else "No relevant context found."
            except Exception as e:
                return f"Error retrieving context: {str(e)}"
        
        # Define the tool
        self.tools = [retrieve_context]
        
        # System prompt for the agent
        system_prompt = """You are a helpful assistant that answers questions based on retrieved document context.

When a user asks a question:
1. Use the retrieve_context tool to get relevant information from the documents
2. Analyze the retrieved context carefully
3. Answer the user's question based on the context provided
4. If the context doesn't contain enough information to answer the question, say so clearly

Always use the retrieve_context tool before answering any question."""
        
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
    """Main function to interact with the NeedleAgent using command line input."""
    print("Initializing NeedleAgent...")
    agent = NeedleAgent()
    print("NeedleAgent ready! Type your questions (or 'quit'/'exit' to stop).\n")
    
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
    
