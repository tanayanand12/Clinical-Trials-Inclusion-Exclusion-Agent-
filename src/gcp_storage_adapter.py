# gcp_storage_adapter.py
import os
import tempfile
import logging
from google.cloud import storage
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class GCPStorageAdapter:
    """Adapter for using Google Cloud Storage with the RAG pipeline."""
    
    def __init__(self, bucket_name: str, credentials_path: Optional[str] = None):
        """
        Initialize GCP Storage adapter.
        
        Args:
            bucket_name: GCS bucket name
            credentials_path: Path to service account credentials (optional)
        """
        if credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
            
        self.bucket_name = bucket_name
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
    
    # def upload_index(self, index_path: str, gcs_path: str) -> bool:
    #     """
    #     Upload index files to GCS.
        
    #     Args:
    #         index_path: Local index path (without extension)
    #         gcs_path: GCS path (without extension)
            
    #     Returns:
    #         Success status
    #     """
    #     try:
    #         # Upload index file
    #         index_blob = self.bucket.blob(f"{gcs_path}.index")
    #         index_blob.upload_from_filename(f"{index_path}.index")
            
    #         # Upload documents file
    #         docs_blob = self.bucket.blob(f"{gcs_path}.documents")
    #         docs_blob.upload_from_filename(f"{index_path}.documents")
            
    #         logger.info(f"Uploaded index to gs://{self.bucket_name}/{gcs_path}")
    #         return True
            
    #     except Exception as e:
    #         logger.error(f"Error uploading index to GCS: {e}")
    #         return False

    def upload_index(self, index_path: str, gcs_path: str) -> bool:
        """
        Upload index files to GCS.
        
        Args:
            index_path: Local index path (without extension)
            gcs_path: GCS path (without extension)
            
        Returns:
            Success status
        """
        try:
            # Upload index file
            index_blob = self.bucket.blob(f"{gcs_path}.index")
            index_blob.upload_from_filename(f"{index_path}.index")
            
            # Upload documents file (check compressed first)
            docs_gz_path = f"{index_path}.documents.gz"
            docs_path = f"{index_path}.documents"
            
            if os.path.exists(docs_gz_path):
                docs_blob = self.bucket.blob(f"{gcs_path}.documents.gz")
                docs_blob.upload_from_filename(docs_gz_path)
            elif os.path.exists(docs_path):
                docs_blob = self.bucket.blob(f"{gcs_path}.documents")
                docs_blob.upload_from_filename(docs_path)
            else:
                raise FileNotFoundError(f"Documents file not found: {docs_path} or {docs_gz_path}")
            
            logger.info(f"Uploaded index to gs://{self.bucket_name}/{gcs_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading index to GCS: {e}")
            return False
        
    def upload_index_to_model_id(self, model_id: str, index_path: str) -> bool:
        """
        Uploads the index files to GCS using model ID.

        Args:
            model_id: Model ID to construct GCS path
            index_path: Local index path (without extension)
        
        Returns:
            Success status
        """
        try:
            # Note: Every index will have name as <model_id>.index and <model_id>.documents
            gcs_path = f"indexes/{model_id}/{model_id}"
            return self.upload_index(index_path, gcs_path)

        except Exception as e:
            logger.error(f"Error uploading index to GCS using model ID {model_id}: {e}")
            return False
    
    # def download_index(self, gcs_path: str, local_path: str) -> bool:
    #     """
    #     Download index files from GCS.
        
    #     Args:
    #         gcs_path: GCS path (without extension)
    #         local_path: Local index path (without extension)
            
    #     Returns:
    #         Success status
    #     """
    #     try:
    #         # Create directory if it doesn't exist
    #         os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
    #         # Download index file
    #         index_blob = self.bucket.blob(f"{gcs_path}.index")
    #         index_blob.download_to_filename(f"{local_path}.index")
            
    #         # Download documents file
    #         docs_blob = self.bucket.blob(f"{gcs_path}.documents")
    #         docs_blob.download_to_filename(f"{local_path}.documents")
            
    #         logger.info(f"Downloaded index from gs://{self.bucket_name}/{gcs_path}")
    #         return True
            
    #     except Exception as e:
    #         logger.error(f"Error downloading index from GCS: {e}")
    #         return False

    def download_index(self, gcs_path: str, local_path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download index file
            index_blob = self.bucket.blob(f"{gcs_path}.index")
            index_blob.download_to_filename(f"{local_path}.index")
            
            # Try compressed first, then uncompressed
            docs_gz_blob = self.bucket.blob(f"{gcs_path}.documents.gz") 
            docs_blob = self.bucket.blob(f"{gcs_path}.documents")
            
            if docs_gz_blob.exists():
                docs_gz_blob.download_to_filename(f"{local_path}.documents.gz")
            elif docs_blob.exists():
                docs_blob.download_to_filename(f"{local_path}.documents")
            else:
                raise FileNotFoundError("No documents file found in GCS")
                
            return True
        except Exception as e:
            logger.error(f"Error downloading index from GCS: {e}")
            return False
        
    def download_index_using_model_id(self, model_id: str, local_path: str) -> bool:
        """
        Download index files from GCS using model ID.
        
        Args:
            model_id: Model ID to construct GCS path
            local_path: Local index path (without extension)
            
        Returns:
            Success status
        """
        try:
            gcs_path = f"indexes/{model_id}/{model_id}"
            return self.download_index(gcs_path, local_path)

        except Exception as e:
            logger.error(f"Error downloading index from GCS using model ID {model_id}: {e}")
            return False
    
    def list_pdfs(self, prefix: str) -> List[str]:
        """
        List PDF files in a GCS directory.
        
        Args:
            prefix: GCS prefix/directory
            
        Returns:
            List of PDF blob names
        """
        try:
            logger.info(f"Listing blobs in GCS bucket {self.bucket_name} with prefix {prefix}")
            blobs = self.bucket.list_blobs(prefix=prefix)
            return [blob.name for blob in blobs if blob.name.endswith('.pdf')]
            
        except Exception as e:
            logger.error(f"Error listing PDFs in GCS: {e}")
            return []
    
    def download_pdfs_to_temp(self, gcs_prefix: str) -> str:
        """
        Download PDFs from GCS to a temporary directory.
        
        Args:
            gcs_prefix: GCS prefix/directory
            
        Returns:
            Path to temporary directory
        """
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp(prefix="rag_pdfs_")
            
            # List and download PDFs
            pdf_blobs = self.list_pdfs(gcs_prefix)
            for blob_name in pdf_blobs:
                local_path = os.path.join(temp_dir, os.path.basename(blob_name))
                self.bucket.blob(blob_name).download_to_filename(local_path)
            
            logger.info(f"Downloaded {len(pdf_blobs)} PDFs to {temp_dir}")
            return temp_dir
            
        except Exception as e:
            logger.error(f"Error downloading PDFs from GCS: {e}")
            return ""
        
    def download_pdfs_to_temp_using_model_id(self, model_id: str) -> str:
        """
        Download PDFs from GCS to a temporary directory using model ID.
        
        Args:
            model_id: Model ID to construct GCS prefix
            
        Returns:
            Path to temporary directory
        """
        try:
            logger.info(f"Downloading PDFs for model ID {model_id}")


            gcs_prefix = f"localDatasets/{model_id}/"
            local_folder = self.download_pdfs_to_temp(gcs_prefix)

            if not local_folder:
                logger.error(f"Failed to download PDFs for model ID {model_id}")
                return ""
            
            return local_folder
            
        except Exception as e:
            logger.error(f"Error downloading PDFs using model ID {model_id}: {e}")
            return ""
