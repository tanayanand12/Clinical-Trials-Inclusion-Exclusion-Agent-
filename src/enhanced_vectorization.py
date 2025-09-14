# enhanced_vectorization.py (replaces vectorization.py)
import os
import time
import logging
import numpy as np #type: ignore
from typing import List, Dict, Any, Optional
import backoff #type: ignore
from dotenv import load_dotenv  #type: ignore
import openai #type: ignore
from functools import lru_cache

logger = logging.getLogger(__name__)

class VectorizationModule:
    """Enhanced module for embedding document content and queries using OpenAI embeddings."""
    
    # Model dimension mapping
    MODEL_DIMENSIONS = {
        "text-embedding-ada-002": 1536,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072
    }
    
    def __init__(self, openai_api_key: str = None, embedding_model: str = "text-embedding-3-large"):
        """
        Initialize the vectorization module.
        
        Args:
            openai_api_key: OpenAI API key
            embedding_model: Embedding model name
        """
        load_dotenv()
        self.embedding_model = embedding_model
        self.embedding_dim = self.MODEL_DIMENSIONS.get(embedding_model, 1536)
        
        # Get API key
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found")
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=self.api_key)
        logger.info(f"Using OpenAI embedding model: {self.embedding_model} (dim: {self.embedding_dim})")
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of current embedding model."""
        return self.embedding_dim
    
    def auto_select_model_by_dimension(self, target_dim: int) -> str:
        """
        Auto-select embedding model based on target dimension.
        
        Args:
            target_dim: Target embedding dimension
            
        Returns:
            Best matching model name
        """
        for model, dim in self.MODEL_DIMENSIONS.items():
            if dim == target_dim:
                return model
        
        # Default fallback
        return "text-embedding-3-small"
    
    @lru_cache(maxsize=1000)
    @backoff.on_exception(
        backoff.expo,
        (openai.RateLimitError, openai.APIError, openai.InternalServerError),
        max_tries=5,
        factor=2
    )
    def get_embedding(self, text: str, model: Optional[str] = None) -> np.ndarray:
        """
        Get an embedding for a single text string with caching.
        
        Args:
            text: Text to embed
            model: Optional model override
            
        Returns:
            Embedding as numpy array
        """
        model = model or self.embedding_model
        
        response = self.client.embeddings.create(
            input=[text],
            model=model
        )
        embedding = response.data[0].embedding
        return np.array(embedding, dtype=np.float32)
    
    @backoff.on_exception(
        backoff.expo,
        (openai.RateLimitError, openai.APIError, openai.InternalServerError),
        max_tries=5,
        factor=2
    )
    def get_batch_embeddings(
        self, 
        texts: List[str], 
        batch_size: int = 256,
        model: Optional[str] = None
    ) -> List[np.ndarray]:
        """
        Get embeddings for a batch of texts with chunking and backoff.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process per API call
            model: Optional model override
            
        Returns:
            List of numpy arrays
        """
        model = model or self.embedding_model
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=model
                )
                
                batch_embeddings = [
                    np.array(item.embedding, dtype=np.float32) 
                    for item in response.data
                ]
                all_embeddings.extend(batch_embeddings)
                
                if i + batch_size < len(texts):
                    time.sleep(0.1)  # Rate limiting
                    
            except openai.BadRequestError as e:
                # Handle context length errors by reducing batch size
                if "maximum context length" in str(e).lower():
                    logger.warning(f"Context length error, reducing batch size from {batch_size}")
                    smaller_batch_size = max(1, batch_size // 2)
                    smaller_embeddings = self.get_batch_embeddings(
                        batch, 
                        batch_size=smaller_batch_size,
                        model=model
                    )
                    all_embeddings.extend(smaller_embeddings)
                else:
                    raise
        
        return all_embeddings
    
    def get_query_vector_auto_dim(self, query: str, target_dim: int) -> np.ndarray:
        """
        Get query embedding with auto model selection based on dimension.
        
        Args:
            query: Query string
            target_dim: Target dimension to match
            
        Returns:
            Query embedding
        """
        model = self.auto_select_model_by_dimension(target_dim)
        return self.get_embedding(query, model=model)
    
    def get_text_embeddings_auto_dim(
        self, 
        texts: List[str], 
        target_dim: int,
        batch_size: int = 256
    ) -> List[np.ndarray]:
        """
        Get text embeddings with auto model selection based on dimension.
        
        Args:
            texts: List of texts to embed
            target_dim: Target dimension to match
            batch_size: Batch size for processing
            
        Returns:
            List of embeddings
        """
        model = self.auto_select_model_by_dimension(target_dim)
        return self.get_batch_embeddings(texts, batch_size=batch_size, model=model)
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Embed a list of document chunks.
        
        Args:
            chunks: List of document chunk dictionaries
            
        Returns:
            List of document chunks with embeddings added
        """
        texts = [chunk['text'] for chunk in chunks]
        if not texts:
            return []
        
        try:
            embeddings = self.get_batch_embeddings(texts)
            
            for i, embedding in enumerate(embeddings):
                chunks[i]['embedding'] = embedding
            
            logger.info(f"Successfully embedded {len(chunks)} document chunks")
            return chunks
        
        except Exception as e:
            logger.error(f"Error embedding chunks: {e}")
            return []
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query string using current model.
        
        Args:
            query: Query string to embed
            
        Returns:
            Query embedding as numpy array
        """
        try:
            return self.get_embedding(query)
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def get_query_vector(self, query: str) -> np.ndarray:
        """Alias for embed_query for backward compatibility."""
        return self.embed_query(query)