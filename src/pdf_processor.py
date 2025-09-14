# pdf_processor.py
import os
import pymupdf4llm # type: ignore
from typing import List, Dict, Any
from .local_generalization_agent import LocalGeneralizationAgent

class PDFProcessor:
    """Module for processing PDF documents and extracting chunked text with metadata."""
    
    def __init__(self, max_char_limit: int = 1000, overlap_percentage: float = 0.2):
        """
        Initialize the PDF processor.
        
        Args:
            max_char_limit: Maximum characters per chunk
            overlap_percentage: Percentage of overlap between chunks
        """
        self.max_char_limit = max_char_limit
        self.overlap_percentage = overlap_percentage
        self.generalization_agent = LocalGeneralizationAgent()  # Initialize agent
    
    def process_folder(self, folder_path: str) -> List[Dict[str, Any]]:
        """
        Process all PDF files in a folder.
        
        Args:
            folder_path: Path to the folder containing PDF files
            
        Returns:
            List of dictionaries containing chunked text and metadata
        """
        all_chunks = []
        
        for pdf_file in os.listdir(folder_path):
            if pdf_file.endswith('.pdf'):
                pdf_path = os.path.join(folder_path, pdf_file)
                try:
                    chunks = self.process_pdf(pdf_path, pdf_file)
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"Error processing {pdf_file}: {e}")
        
        return all_chunks
    
    def process_pdf(self, pdf_path: str, pdf_filename: str) -> List[Dict[str, Any]]:
        """
        Process a single PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            pdf_filename: Name of the PDF file
            
        Returns:
            List of dictionaries containing chunked text and metadata
        """
        chunks = []
        previous_chunk = None
        
        # Extract markdown chunks from the PDF
        page_data = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
        
        for page_number, content in enumerate(page_data):
            text = content['text']
            metadata = {
                'page_number': page_number + 1,
                'toc_items': content.get('toc_items', []),
                'pdf_name': pdf_filename,
                'page_summary': self.generalization_agent.local_generalise(text)
            }
            
            # Assign topic based on headers in the text
            metadata['topic'] = self._extract_topic(text)
            
            # Perform chunking based on headers and markdown formatting
            page_chunks = self._chunk_text(text, metadata)
            
            # Handle header continuation across pages
            if page_number > 0 and self._is_header(content['text'].splitlines()[0]):
                if previous_chunk and page_chunks:
                    # Merge previous chunk with the first chunk of the current page
                    previous_chunk['text'] += "\n" + page_chunks[0]['text']
                    chunks[-1] = previous_chunk  # Update last chunk
                    page_chunks.pop(0)  # Remove first chunk to avoid duplication
            
            chunks.extend(page_chunks)
            previous_chunk = page_chunks[-1] if page_chunks else None
        
        return chunks
    
    def _extract_topic(self, text: str) -> str:
        """
        Extract topic from headers in the text.
        
        Args:
            text: The text to extract the topic from
            
        Returns:
            Extracted topic
        """
        lines = text.splitlines()
        # Check for markdown headers (ATX-style)
        for line in lines:
            if self._is_header(line):
                return line.strip()  # Return the first header found as the topic
        
        # Fallback to first line if no headers found
        if lines and len(lines) > 0:
            return lines[0].strip()  # Use the first line as the topic
        
        return 'Unknown Topic'  # Default if no content is present
    
    def _is_header(self, line: str) -> bool:
        """
        Determine if a line is a header based on its formatting.
        
        Args:
            line: The line to check
            
        Returns:
            True if the line is a header, False otherwise
        """
        return line.strip() != "" and (line.isupper() or line.startswith("#"))
    
    def _chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk the text based on headers and markdown formatting.
        
        Args:
            text: The text to be chunked
            metadata: Metadata associated with the text
            
        Returns:
            List of dictionaries containing chunked text and metadata
        """
        lines = text.splitlines()
        chunks = []
        current_chunk = []
        
        for line in lines:
            # Check if the line is a header
            if self._is_header(line):
                # If there's already content in the current chunk, save it before starting a new one
                if current_chunk:
                    chunk_text = '\n'.join(current_chunk)
                    if len(chunk_text) > self.max_char_limit:
                        # Break the chunk into smaller ones if it exceeds the limit
                        sub_chunks = self._break_large_chunk(chunk_text, metadata)
                        chunks.extend(sub_chunks)
                    else:
                        chunks.append({'text': chunk_text, **metadata})
                    current_chunk = []  # Reset current chunk
            
            # Add the line to the current chunk
            current_chunk.append(line)
        
        # Add any remaining lines as the last chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            if len(chunk_text) > self.max_char_limit:
                # Break the last chunk into smaller ones if it exceeds the limit
                sub_chunks = self._break_large_chunk(chunk_text, metadata)
                chunks.extend(sub_chunks)
            else:
                chunks.append({'text': chunk_text, **metadata})
        
        return chunks
    
    def _break_large_chunk(self, chunk_text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Break a large chunk into smaller ones with overlap.
        
        Args:
            chunk_text: The text to be broken into smaller chunks
            metadata: Metadata associated with the text
            
        Returns:
            List of dictionaries containing chunked text and metadata
        """
        lines = chunk_text.splitlines()
        sub_chunks = []
        current_chunk = []
        chunk_size = int(self.max_char_limit * (1 - self.overlap_percentage))
        
        for line in lines:
            # Add the line to the current sub-chunk
            current_chunk.append(line)
            
            # If the current sub-chunk exceeds the chunk size, save it and start a new one
            if len('\n'.join(current_chunk)) > chunk_size:
                sub_chunks.append({'text': '\n'.join(current_chunk), **metadata})
                
                # Calculate overlap
                overlap_start = max(0, len(current_chunk) - int(len(current_chunk) * self.overlap_percentage))
                current_chunk = current_chunk[overlap_start:]
        
        # Add any remaining lines as the last sub-chunk
        if current_chunk:
            sub_chunks.append({'text': '\n'.join(current_chunk), **metadata})
        
        return sub_chunks