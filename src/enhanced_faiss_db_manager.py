# # enhanced_faiss_db_manager.py (replaces faiss_db_manager.py)
# import numpy as np #type: ignore
# import faiss #type: ignore
# import pickle
# import json
# import gzip
# import os
# from typing import List, Dict, Any, Optional, Tuple
# import logging
# from pathlib import Path
# from langchain_core.documents import Document #type: ignore

# logger = logging.getLogger(__name__)

# class EnhancedFaissVectorDB:
#     """Enhanced FAISS vector database manager with improved functionality."""
    
#     def __init__(self, dimension: Optional[int] = None):
#         """
#         Initialize the enhanced FAISS vector database.
        
#         Args:
#             dimension: Dimension of vectors (auto-detected if not provided)
#         """
#         self.dimension = dimension
#         self.index = None
#         self.documents = []
#         self.is_populated = False
#         self._metadata = {}
    
#     def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
#         """
#         Add documents to the vector database with automatic dimension detection.
        
#         Args:
#             documents: List of document dictionaries with embeddings
            
#         Returns:
#             Success status
#         """
#         if not documents:
#             logger.warning("No documents provided")
#             return False
        
#         try:
#             # Extract embeddings and detect dimension
#             embeddings = [doc['embedding'] for doc in documents if 'embedding' in doc]
#             if not embeddings:
#                 logger.warning("No embeddings found in documents")
#                 return False
            
#             # Auto-detect dimension if not set
#             if self.dimension is None:
#                 self.dimension = len(embeddings[0])
#                 logger.info(f"Auto-detected embedding dimension: {self.dimension}")
            
#             # Initialize index if needed
#             if self.index is None:
#                 self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
#                 logger.info(f"Initialized FAISS index with dimension {self.dimension}")
            
#             # Normalize embeddings for cosine similarity
#             embeddings_array = np.array(embeddings, dtype=np.float32)
#             faiss.normalize_L2(embeddings_array)
            
#             # Add to FAISS index
#             self.index.add(embeddings_array)
            
#             # Store document metadata (without embeddings to save memory)
#             for doc in documents:
#                 doc_copy = doc.copy()
#                 if 'embedding' in doc_copy:
#                     del doc_copy['embedding']
#                 self.documents.append(doc_copy)
            
#             self.is_populated = True
#             self._metadata = {
#                 'dimension': self.dimension,
#                 'total_documents': len(self.documents),
#                 'index_type': 'IndexFlatIP'
#             }
            
#             logger.info(f"Added {len(documents)} documents to FAISS index")
#             return True
            
#         except Exception as e:
#             logger.error(f"Error adding documents to FAISS index: {e}")
#             return False
    
#     def similarity_search(
#         self, 
#         query_embedding: np.ndarray, 
#         k: int = 5, 
#         metadata_filter: Optional[Dict[str, Any]] = None,
#         score_threshold: Optional[float] = None
#     ) -> Tuple[List[Dict[str, Any]], List[float]]:
#         """
#         Perform similarity search with optional filtering and thresholding.
        
#         Args:
#             query_embedding: Query vector
#             k: Number of results to return
#             metadata_filter: Optional metadata filter
#             score_threshold: Minimum similarity score threshold
            
#         Returns:
#             Tuple of (matching documents, similarity scores)
#         """
#         if not self.is_populated or self.index is None:
#             logger.warning("Index is empty")
#             return [], []
        
#         try:
#             # Normalize query vector for cosine similarity
#             query_vector = np.array([query_embedding], dtype=np.float32)
#             faiss.normalize_L2(query_vector)
            
#             # Search with larger k for filtering
#             search_k = k * 4 if metadata_filter else k
#             scores, indices = self.index.search(query_vector, search_k)
#             scores = scores[0]
#             indices = indices[0]
            
#             # Apply filtering and thresholding
#             results = []
#             result_scores = []
            
#             for i, idx in enumerate(indices):
#                 if idx == -1 or idx >= len(self.documents):
#                     continue
                    
#                 score = float(scores[i])
#                 if score_threshold is not None and score < score_threshold:
#                     continue
                
#                 doc = self.documents[idx]
                
#                 # Apply metadata filtering
#                 if metadata_filter:
#                     if not self._matches_filter(doc, metadata_filter):
#                         continue
                
#                 results.append(doc)
#                 result_scores.append(score)
                
#                 if len(results) >= k:
#                     break
            
#             return results, result_scores
            
#         except Exception as e:
#             logger.error(f"Error performing similarity search: {e}")
#             return [], []
    
#     def get_dimension(self) -> int:
#         """Get the dimension of the vector space."""
#         return self.dimension or 0
    
#     def get_document_count(self) -> int:
#         """Get the total number of documents."""
#         return len(self.documents)
    
#     def save(self, path: str, compress: bool = True) -> bool:
#         """
#         Save the vector database with optional compression.
        
#         Args:
#             path: Path to save (without extension)
#             compress: Whether to compress metadata
            
#         Returns:
#             Success status
#         """
#         try:
#             # Ensure directory exists
#             path_obj = Path(path)
#             path_obj.parent.mkdir(parents=True, exist_ok=True)
            
#             # Save FAISS index
#             faiss.write_index(self.index, f"{path}.index")
            
#             # Save documents metadata
#             docs_path = f"{path}.documents"
#             if compress:
#                 docs_path += ".gz"
#                 with gzip.open(docs_path, 'wt', encoding='utf-8') as f:
#                     json.dump({
#                         'documents': self.documents,
#                         'metadata': self._metadata
#                     }, f, indent=2)
#             else:
#                 with open(docs_path, 'w', encoding='utf-8') as f:
#                     json.dump({
#                         'documents': self.documents,
#                         'metadata': self._metadata
#                     }, f, indent=2)
            
#             logger.info(f"Saved vector database to {path}")
#             return True
            
#         except Exception as e:
#             logger.error(f"Error saving vector database: {e}")
#             return False
    
#     def load(self, path: str) -> bool:
#         """
#         Load vector database with automatic format detection.
        
#         Args:
#             path: Path to load (without extension)
            
#         Returns:
#             Success status
#         """
#         try:
#             # Load FAISS index
#             index_path = f"{path}.index"
#             if not os.path.exists(index_path):
#                 logger.error(f"Index file not found: {index_path}")
#                 return False
            
#             self.index = faiss.read_index(index_path)
#             self.dimension = self.index.d
            
#             # Try to load documents (multiple formats)
#             docs_loaded = False
            
#             # Try compressed JSON first
#             docs_path_gz = f"{path}.documents.gz"
#             if os.path.exists(docs_path_gz):
#                 with gzip.open(docs_path_gz, 'rt', encoding='utf-8') as f:
#                     data = json.load(f)
#                     self.documents = data.get('documents', [])
#                     self._metadata = data.get('metadata', {})
#                     docs_loaded = True
            
#             # Try regular JSON
#             if not docs_loaded:
#                 docs_path = f"{path}.documents"
#                 if os.path.exists(docs_path):
#                     with open(docs_path, 'r', encoding='utf-8') as f:
#                         data = json.load(f)
#                         if isinstance(data, dict):
#                             self.documents = data.get('documents', [])
#                             self._metadata = data.get('metadata', {})
#                         else:
#                             self.documents = data  # Legacy format
#                             self._metadata = {}
#                         docs_loaded = True
            
#             # Try pickle format (legacy)
#             if not docs_loaded:
#                 pickle_path = f"{path}.documents"
#                 if os.path.exists(pickle_path):
#                     with open(pickle_path, 'rb') as f:
#                         self.documents = pickle.load(f)
#                         self._metadata = {}
#                         docs_loaded = True
            
#             if not docs_loaded:
#                 logger.error(f"Documents file not found for: {path}")
#                 return False
            
#             self.is_populated = len(self.documents) > 0
#             logger.info(f"Loaded vector database from {path} with {len(self.documents)} documents")
#             return True
            
#         except Exception as e:
#             logger.error(f"Error loading vector database: {e}")
#             return False
    
#     def get_langchain_documents(self, results: List[Dict[str, Any]]) -> List[Document]:
#         """Convert results to LangChain document format."""
#         documents = []
        
#         for doc in results:
#             content = doc.get('text', '')
#             metadata = {k: v for k, v in doc.items() if k != 'text'}
            
#             documents.append(Document(
#                 page_content=content,
#                 metadata=metadata
#             ))
        
#         return documents
    
#     def _matches_filter(self, doc: Dict[str, Any], metadata_filter: Dict[str, Any]) -> bool:
#         """Check if document matches metadata filter."""
#         for key, value in metadata_filter.items():
#             if key not in doc or doc[key] != value:
#                 return False
#         return True
    
#     def get_stats(self) -> Dict[str, Any]:
#         """Get database statistics."""
#         if not self.is_populated:
#             return {'status': 'empty'}
        
#         return {
#             'status': 'populated',
#             'dimension': self.dimension,
#             'document_count': len(self.documents),
#             'index_type': self._metadata.get('index_type', 'unknown'),
#             'metadata': self._metadata
#         }



# enhanced_faiss_db_manager.py (replaces faiss_db_manager.py)
import numpy as np #type: ignore
import faiss #type: ignore
import pickle
import json
import gzip
import os
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
from langchain_core.documents import Document #type: ignore
import os
import json
import gzip
import pickle
import chardet #type: ignore

logger = logging.getLogger(__name__)

class EnhancedFaissVectorDB:
    """Enhanced FAISS vector database manager with improved functionality."""
    
    def __init__(self, dimension: Optional[int] = None):
        """
        Initialize the enhanced FAISS vector database.
        
        Args:
            dimension: Dimension of vectors (auto-detected if not provided)
        """
        self.dimension = dimension
        self.index = None
        self.documents = []
        self.is_populated = False
        self._metadata = {}
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Add documents to the vector database with automatic dimension detection.
        
        Args:
            documents: List of document dictionaries with embeddings
            
        Returns:
            Success status
        """
        if not documents:
            logger.warning("No documents provided")
            return False
        
        try:
            # Extract embeddings and detect dimension
            embeddings = [doc['embedding'] for doc in documents if 'embedding' in doc]
            if not embeddings:
                logger.warning("No embeddings found in documents")
                return False
            
            # Auto-detect dimension if not set
            if self.dimension is None:
                self.dimension = len(embeddings[0])
                logger.info(f"Auto-detected embedding dimension: {self.dimension}")
            
            # Initialize index if needed
            if self.index is None:
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
                logger.info(f"Initialized FAISS index with dimension {self.dimension}")
            
            # Normalize embeddings for cosine similarity
            embeddings_array = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(embeddings_array)
            
            # Add to FAISS index
            self.index.add(embeddings_array)
            
            # Store document metadata (without embeddings to save memory)
            for doc in documents:
                doc_copy = doc.copy()
                if 'embedding' in doc_copy:
                    del doc_copy['embedding']
                self.documents.append(doc_copy)
            
            self.is_populated = True
            self._metadata = {
                'dimension': self.dimension,
                'total_documents': len(self.documents),
                'index_type': 'IndexFlatIP'
            }
            
            logger.info(f"Added {len(documents)} documents to FAISS index")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to FAISS index: {e}")
            return False
    
    def similarity_search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 5, 
        metadata_filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Perform similarity search with optional filtering and thresholding.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            metadata_filter: Optional metadata filter
            score_threshold: Minimum similarity score threshold
            
        Returns:
            Tuple of (matching documents, similarity scores)
        """
        if not self.is_populated or self.index is None:
            logger.warning("Index is empty")
            return [], []
        
        try:
            # Normalize query vector for cosine similarity
            query_vector = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_vector)
            
            # Search with larger k for filtering
            search_k = k * 4 if metadata_filter else k
            scores, indices = self.index.search(query_vector, search_k)
            scores = scores[0]
            indices = indices[0]
            
            # Apply filtering and thresholding
            results = []
            result_scores = []
            
            for i, idx in enumerate(indices):
                if idx == -1 or idx >= len(self.documents):
                    continue
                    
                score = float(scores[i])
                if score_threshold is not None and score < score_threshold:
                    continue
                
                doc = self.documents[idx]
                
                # Apply metadata filtering
                if metadata_filter:
                    if not self._matches_filter(doc, metadata_filter):
                        continue
                
                results.append(doc)
                result_scores.append(score)
                
                if len(results) >= k:
                    break
            
            return results, result_scores
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            return [], []
    
    def get_dimension(self) -> int:
        """Get the dimension of the vector space."""
        return self.dimension or 0
    
    def get_document_count(self) -> int:
        """Get the total number of documents."""
        return len(self.documents)
    
    def save(self, path: str, compress: bool = True) -> bool:
        """
        Save the vector database with optional compression.
        
        Args:
            path: Path to save (without extension)
            compress: Whether to compress metadata
            
        Returns:
            Success status
        """
        try:
            # Ensure directory exists
            path_obj = Path(path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, f"{path}.index")
            
            # Save documents metadata
            docs_path = f"{path}.documents"
            if compress:
                docs_path += ".gz"
                with gzip.open(docs_path, 'wt', encoding='utf-8') as f:
                    json.dump({
                        'documents': self.documents,
                        'metadata': self._metadata
                    }, f, indent=2)
            else:
                with open(docs_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'documents': self.documents,
                        'metadata': self._metadata
                    }, f, indent=2)
            
            logger.info(f"Saved vector database to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving vector database: {e}")
            return False
    
    # def load(self, path: str) -> bool:
    #     """
    #     Load vector database with automatic format detection.
        
    #     Args:
    #         path: Path to load (without extension)
            
    #     Returns:
    #         Success status
    #     """
    #     try:
    #         # Load FAISS index
    #         index_path = f"{path}.index"
    #         if not os.path.exists(index_path):
    #             logger.error(f"Index file not found: {index_path}")
    #             return False
            
    #         self.index = faiss.read_index(index_path)
    #         self.dimension = self.index.d
            
    #         # Try to load documents (multiple formats)
    #         docs_loaded = False
            
    #         # Try compressed JSON first
    #         docs_path_gz = f"{path}.documents.gz"
    #         if os.path.exists(docs_path_gz):
    #             with gzip.open(docs_path_gz, 'rt', encoding='utf-8') as f:
    #                 data = json.load(f)
    #                 self.documents = data.get('documents', [])
    #                 self._metadata = data.get('metadata', {})
    #                 docs_loaded = True
            
    #         # Try regular JSON
    #         if not docs_loaded:
    #             docs_path = f"{path}.documents"
    #             if os.path.exists(docs_path):
    #                 with open(docs_path, 'r', encoding='utf-8') as f:
    #                     data = json.load(f)
    #                     if isinstance(data, dict):
    #                         self.documents = data.get('documents', [])
    #                         self._metadata = data.get('metadata', {})
    #                     else:
    #                         self.documents = data  # Legacy format
    #                         self._metadata = {}
    #                     docs_loaded = True
            
    #         # Try pickle format (legacy)
    #         if not docs_loaded:
    #             pickle_path = f"{path}.documents"
    #             if os.path.exists(pickle_path):
    #                 with open(pickle_path, 'rb') as f:
    #                     self.documents = pickle.load(f)
    #                     self._metadata = {}
    #                     docs_loaded = True
            
    #         if not docs_loaded:
    #             logger.error(f"Documents file not found for: {path}")
    #             return False
            
    #         self.is_populated = len(self.documents) > 0
    #         logger.info(f"Loaded vector database from {path} with {len(self.documents)} documents")
    #         return True
            
    #     except Exception as e:
    #         logger.error(f"Error loading vector database: {e}")
    #         return False

    def load(self, path: str) -> bool:
        """
        Load the vector database from disk with fallback for binary pickle format.

        Args:
            path: Path to load the database from

        Returns:
            Success status
        """
        try:
            # Load the FAISS index
            self.index = faiss.read_index(f"{path}.index")

            # Load the documents metadata
            docs_path = f"{path}.documents"
            if not os.path.exists(docs_path):
                logger.error(f"Documents file not found: {docs_path}")
                return False

            try:
                # Try reading as UTF-8 JSON
                with open(docs_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.documents = data.get('documents', [])
                    else:
                        self.documents = data
            except UnicodeDecodeError:
                # Fallback to binary pickle
                with open(docs_path, 'rb') as f:
                    self.documents = pickle.load(f)

            self.is_populated = len(self.documents) > 0
            logger.info(f"Loaded vector database from {path} with {len(self.documents)} documents")
            return True

        except Exception as e:
            logger.error(f"Error loading vector database: {e}")
            return False

    # def detect_encoding(self, path: str) -> str:
    #     with open(path, 'rb') as f:
    #         raw = f.read(2048)
    #         result = chardet.detect(raw)
    #         return result.get('encoding') or 'utf-8'

    # def load(self, path: str) -> bool:
    #     """
    #     Load vector database with automatic format detection and encoding fallback.

    #     Args:
    #         path: Path to load (without extension)

    #     Returns:
    #         Success status
    #     """
    #     try:
    #         # Load FAISS index
    #         index_path = f"{path}.index"
    #         if not os.path.exists(index_path):
    #             logger.error(f"Index file not found: {index_path}")
    #             return False

    #         self.index = faiss.read_index(index_path)
    #         self.dimension = self.index.d

    #         # Load documents
    #         docs_loaded = False

    #         # Try compressed JSON first
    #         docs_path_gz = f"{path}.documents.gz"
    #         if os.path.exists(docs_path_gz):
    #             try:
    #                 with gzip.open(docs_path_gz, 'rt', encoding='utf-8') as f:
    #                     data = json.load(f)
    #                     self.documents = data.get('documents', [])
    #                     self._metadata = data.get('metadata', {})
    #                     docs_loaded = True
    #             except Exception as e:
    #                 logger.warning(f"Failed to load .gz JSON: {e}")

    #         # Try regular .documents file
    #         if not docs_loaded:
    #             docs_path = f"{path}.documents"
    #             if os.path.exists(docs_path):
    #                 try:
    #                     encoding = self.detect_encoding(docs_path)
    #                     with open(docs_path, 'r', encoding=encoding) as f:
    #                         data = json.load(f)
    #                         if isinstance(data, dict):
    #                             self.documents = data.get('documents', [])
    #                             self._metadata = data.get('metadata', {})
    #                         else:
    #                             self.documents = data
    #                             self._metadata = {}
    #                         docs_loaded = True
    #                 except Exception as json_error:
    #                     logger.warning(f"Failed to load JSON with encoding fallback: {json_error}")
    #                     try:
    #                         with open(docs_path, 'rb') as f:
    #                             self.documents = pickle.load(f)
    #                             self._metadata = {}
    #                             docs_loaded = True
    #                     except Exception as pickle_error:
    #                         logger.error(f"Failed to load pickle fallback: {pickle_error}")

    #         if not docs_loaded:
    #             logger.error(f"Documents file not found or unreadable for: {path}")
    #             return False

    #         self.is_populated = len(self.documents) > 0
    #         logger.info(f"Loaded vector database from {path} with {len(self.documents)} documents")
    #         return True

    #     except Exception as e:
    #         logger.error(f"Error loading vector database: {e}")
    #         return False
    
    def get_langchain_documents(self, results: List[Dict[str, Any]]) -> List[Document]:
        """Convert results to LangChain document format."""
        documents = []
        
        for doc in results:
            content = doc.get('text', '')
            metadata = {k: v for k, v in doc.items() if k != 'text'}
            
            documents.append(Document(
                page_content=content,
                metadata=metadata
            ))
        
        return documents
    
    def _matches_filter(self, doc: Dict[str, Any], metadata_filter: Dict[str, Any]) -> bool:
        """Check if document matches metadata filter."""
        for key, value in metadata_filter.items():
            if key not in doc or doc[key] != value:
                return False
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not self.is_populated:
            return {'status': 'empty'}
        
        return {
            'status': 'populated',
            'dimension': self.dimension,
            'document_count': len(self.documents),
            'index_type': self._metadata.get('index_type', 'unknown'),
            'metadata': self._metadata
        }