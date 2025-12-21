"""Main entry point for interactive querying using Query Router Agent."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents import QueryRouterAgent, RetrievalStrategy


def main():
    """Interactive query interface for Query Router Agent."""
    print("=" * 60)
    print("Query Router Agent - Interactive Query Interface")
    print("=" * 60)
    print("\nInitializing Query Router Agent...")
    
    try:
        router = QueryRouterAgent()
        print("Query Router Agent initialized successfully!\n")
    except Exception as e:
        print(f"Error initializing Query Router Agent: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("-" * 60)
    print("Enter your queries below. Type 'exit' or 'quit' to stop.")
    print("The agent will automatically route your query to the appropriate retrieval strategy:")
    print("  - Chunks: For specific details and precise information")
    print("  - Summaries: For high-level overviews")
    print("-" * 60)
    
    while True:
        try:
            # Get query from user
            query = input("\nQuery: ").strip()
            
            # Check for exit commands
            if query.lower() in ['exit', 'quit', 'q']:
                print("\nExiting. Goodbye!")
                break
            
            if not query:
                print("Please enter a valid query.")
                continue
            
            # Route the query using the router agent
            print(f"\nAnalyzing query: '{query}'")
            print("-" * 60)
            
            router.answer_with_contexts(query)
            
            print("\n" + "-" * 60)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"\nError processing query: {e}")
            import traceback
            traceback.print_exc()
            print("\nPlease try again or type 'exit' to quit.")


if __name__ == "__main__":
    main()

