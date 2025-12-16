"""Main entry point for interactive querying using Auto-Merging Retriever."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval import AutoMergingRetriever


def main():
    """Interactive query interface for Auto-Merging Retriever."""
    print("=" * 60)
    print("Auto-Merging Retriever - Interactive Query Interface")
    print("=" * 60)
    print("\nInitializing retriever...")
    
    try:
        retriever = AutoMergingRetriever()
        print("Retriever initialized successfully!")
        
        # Initialize query engine for synthesized answers
        print("Initializing query engine...")
        query_engine = retriever.get_query_engine()
        print("Query engine initialized successfully!\n")
    except Exception as e:
        print(f"Error initializing retriever/query engine: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("-" * 60)
    print("Enter your queries below. Type 'exit' or 'quit' to stop.")
    print("You will receive synthesized answers based on retrieved documents.")
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
            
            # Query using the query engine
            print(f"\nProcessing query: '{query}'")
            print("Retrieving relevant documents and generating answer...")
            print("-" * 60)
            
            response = query_engine.query(query)
        
            
            # Optionally display source nodes
            if hasattr(response, 'source_nodes') and response.source_nodes:
                print(f"\nSources ({len(response.source_nodes)} document(s) used):")
                for i, node_with_score in enumerate(response.source_nodes, 1):
                    print(f"\n  Source {i}:")
                    if hasattr(node_with_score, 'score'):
                        print(f"    Relevance Score: {node_with_score.score:.4f}")
                    
                    # Extract the actual node from NodeWithScore
                    node = node_with_score.node if hasattr(node_with_score, 'node') else node_with_score
                    
                    metadata = node.metadata if hasattr(node, 'metadata') else {}
                    if metadata:
                        claim_id = metadata.get('claim_id', 'N/A')
                        source_file = metadata.get('source_file', 'N/A')
                        print(f"    Claim ID: {claim_id}")
                        print(f"    Source File: {source_file}")
                    
                    # Show preview of source text
                    text = node.get_content() if hasattr(node, 'get_content') else str(node)
                    preview_length = 200
                    if len(text) > preview_length:
                        print(f"    Preview: {text[:preview_length]}...")
                    else:
                        print(f"    Content: {text}")
            
            # Display synthesized answer
            print("\nAnswer:")
            print("-" * 60)
            print(response.response)
            print("-" * 60)
            
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

