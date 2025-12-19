"""Summary Agent for retrieving context from summaries and answering broad questions."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent

from src.utils.config_loader import load_config, get_vector_store_config
from src.utils.vector_store import ChromaDBManager


class SummaryAgent:
    """
    Simple LangChain agent that uses summaries retrieval to answer broad user queries.
    The agent has a single tool called retrieve_context that retrieves relevant
    summaries, then uses that context to answer the user's question.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize Summary Agent.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path)
        llm_config = self.config.get('llm', {})
        vector_store_config = get_vector_store_config(self.config)
        
        # Initialize ChromaDB manager for summaries collection
        self.summary_manager = ChromaDBManager(
            persist_directory=vector_store_config['persist_directory'],
            collection_name=vector_store_config['collection_summaries'],
            embedding_model_name=vector_store_config['embedding_model']
        )
        
        # Get the vector store index for summaries
        self.index = self.summary_manager.get_index()
        
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
        
        # Create the retrieve_context tool using closure to access self.index
        @tool
        def retrieve_context(query: str) -> str:
            """Retrieve relevant context from document summaries using vector similarity search."""
            try:
                # Create a simple retriever from the index
                retriever = self.index.as_retriever(similarity_top_k=default_top_k)
                
                # Retrieve nodes
                nodes = retriever.retrieve(query)
                
                # Combine all retrieved summaries into a single string
                context_parts = []
                for node_with_score in nodes:
                    node = node_with_score.node
                    summary_text = node.get_content()
                    context_parts.append(summary_text)
                
                context = "\n\n".join(context_parts)
                return context if context else "No relevant summaries found."
            except Exception as e:
                return f"Error retrieving summaries: {str(e)}"
        
        # Define the tool
        self.tools = [retrieve_context]
        
        # System prompt for the agent
        system_prompt = """You are a helpful assistant that answers broad, high-level questions based on retrieved document summaries.

When a user asks a question:
1. Use the retrieve_context tool to get relevant summaries from the documents
2. Analyze the retrieved summaries carefully
3. Answer the user's question based on the summary context provided
4. Focus on providing high-level overviews, timelines, key events, and general understanding
5. If the summaries don't contain enough information to answer the question, say so clearly

Always use the retrieve_context tool before answering any question.
The summaries provide high-level overviews, so use them to answer broad questions about document content, timelines, key events, and general understanding."""
        
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
        # Invoke the agent with the user query
        for event in self.agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
        ):
            event["messages"][-1].pretty_print()
    
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

