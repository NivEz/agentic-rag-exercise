"""Response Collector for tracking agent responses and retrieved contexts."""

from typing import List


class ResponseCollector:
    """
    Collects responses and contexts from agent execution.
    Used for evaluation and debugging purposes.
    """
    
    def __init__(self):
        """Initialize the collector with empty lists."""
        self.text_response = ""
        self.contexts: List[str] = []
    
    def collect_response(self, response: str) -> None:
        """
        Collect a text response from the agent.
        
        Args:
            response: The text response to collect
        """
        self.text_response = response
    
    def collect_contexts(self, contexts: List[str]) -> None:
        """
        Collect retrieved contexts.
        
        Args:
            contexts: List of context strings to collect
        """
        self.contexts.extend(contexts)
    
    def get_result(self) -> dict:
        """
        Get the collected results.
        
        Returns:
            Dictionary with 'text_response' and 'contexts' keys
        """
        return {
            'text_response': self.text_response,
            'contexts': self.contexts.copy()
        }
    
    def reset(self) -> None:
        """Reset the collector for a new query."""
        self.text_response = ""
        self.contexts.clear()
