# config.py
"""
Configuration module for Clinical Trials RAG System
"""
from typing import Dict, Any
import os
from dotenv import load_dotenv # type: ignore

load_dotenv()

class RAGConfig:
    """Configuration class for RAG system settings"""
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4")
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # Fast and efficient
    EMBEDDING_DIMENSION: int = 384  # Dimension for all-MiniLM-L6-v2
    
    # Chunking Configuration
    CHUNK_SIZE: int = 512  # Optimal for sentence transformers
    CHUNK_OVERLAP: int = 50  # Small overlap for context continuity
    
    # Retrieval Configuration
    TOP_K_CHUNKS: int = 5  # Number of chunks to retrieve
    SIMILARITY_THRESHOLD: float = 0.7  # Minimum similarity score
    
    # Vector Store Configuration
    INDEX_TYPE: str = "IndexFlatIP"  # Inner product for cosine similarity
    NORMALIZE_VECTORS: bool = True  # Normalize for cosine similarity
    
    # System Configuration
    MAX_RETRIES: int = 3
    REQUEST_TIMEOUT: int = 30
    BATCH_SIZE: int = 32  # For efficient embedding
    
    # Clinical Trials API Configuration
    MAX_STUDIES_PER_QUERY: int = 100
    API_TIMEOUT: int = 30
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get all configuration as dictionary"""
        return {
            attr: getattr(cls, attr) 
            for attr in dir(cls) 
            if not attr.startswith('_') and not callable(getattr(cls, attr))
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate essential configuration"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        return True


# Initialize and validate configuration
RAGConfig.validate_config()