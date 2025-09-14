# clinical_trials_vectorizer.py
import os
import time
import logging
import numpy as np # type: ignore
from typing import List, Dict, Any
import backoff # type: ignore
from dotenv import load_dotenv # type: ignore
from openai import OpenAI, APIError, APITimeoutError, RateLimitError, APIConnectionError, BadRequestError # type: ignore
from openai.types.create_embedding_response import CreateEmbeddingResponse # type: ignore
from openai.types import Embedding # type: ignore

logger = logging.getLogger(__name__)

class ClinicalTrialsVectorizer:
    """
    Module for embedding clinical trial chunks and queries using OpenAI API.
    Optimized for clinical trial data with appropriate chunking and batch processing.
    """

    def __init__(self, openai_model: str = "text-embedding-ada-002"):
        """
        Initialize the vectorization module.
        
        Args:
            openai_model: The OpenAI embedding model to use
        """
        load_dotenv()
        self.openai_model = openai_model
        self.embedding_dim = 1536  # text-embedding-ada-002 returns 1536-dimensional embeddings
        
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        logger.info(f"Initialized ClinicalTrialsVectorizer with model: {self.openai_model}")

    @backoff.on_exception(
        backoff.expo,
        (APIError, APITimeoutError, RateLimitError, APIConnectionError, BadRequestError, Exception),
        max_tries=5,
        factor=2
    )
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get an embedding for a single text string.
        
        Args:
            text: Text to embed
        
        Returns:
            numpy array representing the embedding
        """
        try:
            # Ensure text is not empty
            if not text or not text.strip():
                logger.warning("Empty text provided for embedding, returning zero vector")
                return np.zeros(self.embedding_dim)
                
            # Truncate text if too long (OpenAI has token limits)
            max_tokens = 8000  # Conservative limit
            if len(text) > max_tokens:
                text = text[:max_tokens]
                logger.warning(f"Truncated text to {max_tokens} characters for embedding")
            
            response: CreateEmbeddingResponse = self.client.embeddings.create(
                input=[text],
                model=self.openai_model,
                encoding_format="float"
            )
            
            embedding: Embedding = response.data[0]
            return np.array(embedding.embedding)
            
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return np.zeros(self.embedding_dim)

    @backoff.on_exception(
        backoff.expo,
        (APIError, APITimeoutError, RateLimitError, APIConnectionError, BadRequestError, Exception),
        max_tries=5,
        factor=2
    )
    def get_batch_embeddings(self, texts: List[str], batch_size: int = 50) -> List[np.ndarray]:
        """
        Get embeddings for a batch of texts with rate limiting.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process per API call
        
        Returns:
            List of numpy arrays representing embeddings
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Filter out empty texts
            valid_batch = []
            valid_indices = []
            for j, text in enumerate(batch):
                if text and text.strip():
                    valid_batch.append(text[:8000])  # Truncate if needed
                    valid_indices.append(j)
                    
            if not valid_batch:
                # All texts in batch were empty
                batch_embeddings = [np.zeros(self.embedding_dim) for _ in batch]
                all_embeddings.extend(batch_embeddings)
                continue
            
            try:
                response: CreateEmbeddingResponse = self.client.embeddings.create(
                    input=valid_batch,
                    model=self.openai_model,
                    encoding_format="float"
                )
                
                # Create embeddings array for full batch (including empty texts)
                batch_embeddings = []
                valid_idx = 0
                
                for j in range(len(batch)):
                    if j in valid_indices:
                        batch_embeddings.append(np.array(response.data[valid_idx].embedding))
                        valid_idx += 1
                    else:
                        batch_embeddings.append(np.zeros(self.embedding_dim))
                
                all_embeddings.extend(batch_embeddings)
                
                # Rate limiting - pause between batches
                if i + batch_size < len(texts):
                    time.sleep(0.5)
                    
            except Exception as e:
                logger.error(f"Error in batch embedding: {e}")
                # Return zero vectors for failed batch
                batch_embeddings = [np.zeros(self.embedding_dim) for _ in batch]
                all_embeddings.extend(batch_embeddings)
        
        logger.info(f"Successfully embedded {len(all_embeddings)} texts")
        return all_embeddings

    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Embed clinical trial chunks.
        
        Args:
            chunks: List of chunk dictionaries containing 'content' and metadata
        
        Returns:
            Dictionary mapping chunk IDs to embeddings
        """
        if not chunks:
            logger.warning("No chunks provided for embedding")
            return {}
        
        # Extract content and create chunk IDs
        chunk_texts = []
        chunk_ids = []
        
        for i, chunk in enumerate(chunks):
            content = chunk.get("content", "")
            chunk_id = f"{chunk.get('study_id', 'unknown')}_{chunk.get('chunk_type', 'unknown')}_{i}"
            
            chunk_texts.append(content)
            chunk_ids.append(chunk_id)
        
        # Get embeddings
        embeddings = self.get_batch_embeddings(chunk_texts)
        
        # Create mapping
        embedded_chunks = {}
        for chunk_id, embedding, chunk in zip(chunk_ids, embeddings, chunks):
            embedded_chunks[chunk_id] = {
                'embedding': embedding,
                'metadata': chunk
            }
        
        logger.info(f"Successfully embedded {len(embedded_chunks)} clinical trial chunks")
        return embedded_chunks

    def embed_query(self, query_text: str) -> np.ndarray:
        """
        Embed a query string for similarity search.
        
        Args:
            query_text: The query text to embed
        
        Returns:
            numpy array representing the query embedding
        """
        try:
            if not query_text or not query_text.strip():
                logger.warning("Empty query provided")
                return np.zeros(self.embedding_dim)
                
            return self.get_embedding(query_text)
            
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            return np.zeros(self.embedding_dim)

    def compute_similarity(self, query_embedding: np.ndarray, chunk_embeddings: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute cosine similarity between query and chunk embeddings.
        
        Args:
            query_embedding: Query embedding vector
            chunk_embeddings: Dictionary mapping chunk IDs to embeddings
        
        Returns:
            Dictionary mapping chunk IDs to similarity scores
        """
        similarities = {}
        
        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            logger.warning("Query embedding is zero vector")
            return {chunk_id: 0.0 for chunk_id in chunk_embeddings.keys()}
        
        normalized_query = query_embedding / query_norm
        
        for chunk_id, chunk_data in chunk_embeddings.items():
            embedding = chunk_data['embedding']
            
            # Normalize chunk embedding
            chunk_norm = np.linalg.norm(embedding)
            if chunk_norm == 0:
                similarities[chunk_id] = 0.0
                continue
            
            normalized_chunk = embedding / chunk_norm
            
            # Compute cosine similarity
            similarity = np.dot(normalized_query, normalized_chunk)
            similarities[chunk_id] = float(similarity)
        
        return similarities

    def find_most_similar_chunks(self, 
                                query_embedding: np.ndarray, 
                                chunk_embeddings: Dict[str, np.ndarray],
                                top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find the most similar chunks to a query.
        
        Args:
            query_embedding: Query embedding vector
            chunk_embeddings: Dictionary mapping chunk IDs to embedding data
            top_k: Number of top similar chunks to return
        
        Returns:
            List of chunk dictionaries with similarity scores
        """
        similarities = self.compute_similarity(query_embedding, chunk_embeddings)
        
        # Sort by similarity score (descending)
        sorted_chunks = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        # Get top k chunks with metadata
        top_chunks = []
        for chunk_id, similarity_score in sorted_chunks[:top_k]:
            chunk_data = chunk_embeddings[chunk_id]
            chunk_info = {
                'chunk_id': chunk_id,
                'similarity_score': similarity_score,
                'content': chunk_data['metadata'].get('content', ''),
                'chunk_type': chunk_data['metadata'].get('chunk_type', ''),
                'study_id': chunk_data['metadata'].get('study_id', ''),
                'section': chunk_data['metadata'].get('section', ''),
                'metadata': chunk_data['metadata']
            }
            top_chunks.append(chunk_info)
        
        logger.info(f"Found {len(top_chunks)} most similar chunks out of {len(similarities)} total chunks")
        return top_chunks