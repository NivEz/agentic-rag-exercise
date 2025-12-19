"""Summary generator for creating chunk-level summaries using LLM."""

from typing import List, Optional
from llama_index.core import Settings
from llama_index.core.schema import BaseNode, TextNode
from tqdm import tqdm


class SummaryGenerator:
    """Generates summaries for chunks using LLM via LlamaIndex Settings."""
    
    def __init__(
        self,
        summary_instruction: str = "Summarize the following text, focusing on key information, dates, and important details."
    ):
        """
        Initialize summary generator.
        
        Args:
            summary_instruction: Instruction prompt for summarization
        """
        self.summary_instruction = summary_instruction
    
    def generate_summary_for_chunk(self, chunk: BaseNode) -> TextNode:
        """
        Generate a summary for a single chunk using LLM.
        
        Args:
            chunk: The chunk node to summarize
            
        Returns:
            TextNode containing the summary with metadata
        """
        # Get chunk text
        chunk_text = chunk.get_content()
        
        # Create prompt for summarization
        prompt = f"{self.summary_instruction}\n\nText to summarize:\n{chunk_text}"
        
        # Use LlamaIndex Settings.llm to generate summary
        llm = Settings.llm
        if llm is None:
            raise ValueError("LLM not configured in Settings. Please set Settings.llm before using SummaryGenerator.")
        
        # Generate summary
        response = llm.complete(prompt)
        summary_text = response.text.strip()
        
        # Create summary node with metadata
        summary_node = TextNode(
            text=summary_text,
            metadata={
                **chunk.metadata,  # Preserve original metadata
                'is_summary': True,
                'source_chunk_id': chunk.node_id,
                'chunk_level': chunk.metadata.get('chunk_level', 'unknown')
            }
        )
        
        # Preserve relationships if they exist
        if hasattr(chunk, 'relationships'):
            summary_node.relationships = chunk.relationships.copy()
        
        return summary_node
    
    def generate_summaries(
        self,
        chunks: List[BaseNode],
        show_progress: bool = True
    ) -> List[TextNode]:
        """
        Generate summaries for a list of chunks.
        
        Args:
            chunks: List of chunk nodes to summarize
            show_progress: Whether to show progress bar
            
        Returns:
            List of TextNode objects containing summaries
        """
        summaries = []
        
        # Use tqdm for progress tracking
        iterator = tqdm(chunks, desc="Generating summaries") if show_progress else chunks
        
        for chunk in iterator:
            try:
                summary = self.generate_summary_for_chunk(chunk)
                summaries.append(summary)
            except Exception as e:
                print(f"\nError generating summary for chunk {chunk.node_id}: {e}")
                # Create a fallback summary with the original text truncated
                fallback_text = chunk.get_content()[:200] + "..."
                fallback_node = TextNode(
                    text=fallback_text,
                    metadata={
                        **chunk.metadata,
                        'is_summary': True,
                        'source_chunk_id': chunk.node_id,
                        'chunk_level': chunk.metadata.get('chunk_level', 'unknown'),
                        'summary_error': str(e)
                    }
                )
                summaries.append(fallback_node)
        
        return summaries

