"""Summary generator for creating multi-level summaries using LLM."""

from typing import List, Optional, Dict
from llama_index.core import Settings, Document
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.node_parser import SentenceSplitter
from tqdm import tqdm


class SummaryGenerator:
    """Generates summaries using MapReduce approach with SentenceSplitter."""
    
    # System prompt for summarization
    SUMMARY_INSTRUCTION = (
        "Summarize this text with a focus on: "
        "1. Key entities (people, organizations, locations). "
        "2. A timeline of events (dates and actions). "
        "3. Main ideas. "
        "Output a concise summary paragraph."
    )
    
    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 200
    ):
        """
        Initialize summary generator.
        
        Args:
            chunk_size: Chunk size for SentenceSplitter
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
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
        prompt = f"{self.SUMMARY_INSTRUCTION}\n\nText to summarize:\n{chunk_text}"
        
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
    
    def generate_summary_from_summaries(
        self,
        summaries: List[str],
        metadata: Optional[Dict] = None
    ) -> TextNode:
        """
        Generate a document-level summary from a list of chunk summaries.
        
        Args:
            summaries: List of summary texts to combine
            metadata: Metadata to attach to the summary node
            
        Returns:
            TextNode containing the document-level summary
        """
        # Combine summaries into one text
        combined_text = "\n\n".join(summaries)
        
        # Create prompt for summarization
        prompt = f"Summarize the following summaries into a comprehensive document overview. {self.SUMMARY_INSTRUCTION}\n\nSummaries to combine:\n{combined_text}"
        
        # Use LlamaIndex Settings.llm to generate summary
        llm = Settings.llm
        if llm is None:
            raise ValueError("LLM not configured in Settings.")
        
        # Generate summary
        response = llm.complete(prompt)
        summary_text = response.text.strip()
        
        # Create summary node with metadata
        summary_node = TextNode(
            text=summary_text,
            metadata={
                **(metadata or {}),
                'is_summary': True,
                'summary_level': 'document'
            }
        )
        
        return summary_node
    
    def generate_summaries_from_document(
        self,
        document: Document,
        show_progress: bool = True
    ) -> Dict[str, List[TextNode]]:
        """
        Generate summaries from a document using MapReduce approach.
        
        Map: Split document text with SentenceSplitter, summarize each chunk
        Reduce: Combine all chunk summaries into one document summary
        
        Args:
            document: Document to generate summaries for
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary with keys 'chunk' and 'document' containing summaries
        """
        result = {
            'chunk': [],
            'document': []
        }
        
        claim_id = document.metadata.get('claim_id', 'unknown')
        print(f"  Processing document: {claim_id}")
        
        # MAP: Split text using SentenceSplitter and summarize each chunk
        print(f"    MAP: Splitting text with SentenceSplitter (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})...")
        text_chunks = self.splitter.get_nodes_from_documents([document])
        print(f"    Created {len(text_chunks)} text chunks")
        
        print(f"    MAP: Summarizing {len(text_chunks)} chunks...")
        chunk_summaries = []
        for i, text_chunk in enumerate(tqdm(text_chunks, desc="Chunk summaries", disable=not show_progress)):
            try:
                summary = self.generate_summary_for_chunk(text_chunk)
                # Add chunk-specific metadata
                summary.metadata['summary_level'] = 'chunk'
                summary.metadata['chunk_index'] = i
                chunk_summaries.append(summary)
            except Exception as e:
                print(f"\n    Error summarizing chunk {i}: {e}")
        
        result['chunk'].extend(chunk_summaries)
        print(f"    Generated {len(chunk_summaries)} chunk summaries")
        
        # REDUCE: Combine all chunk summaries into document summary
        if chunk_summaries:
            print(f"    REDUCE: Combining {len(chunk_summaries)} chunk summaries into document summary...")
            summary_texts = [s.get_content() for s in chunk_summaries]
            
            try:
                # Prepare metadata with document_id
                doc_metadata = {
                    **document.metadata,
                    'document_id': document.doc_id
                }
                doc_summary = self.generate_summary_from_summaries(
                    summary_texts,
                    metadata=doc_metadata
                )
                result['document'].append(doc_summary)
                print(f"    Created document summary")
            except Exception as e:
                print(f"\n    Error generating document summary: {e}")
        
        return result

