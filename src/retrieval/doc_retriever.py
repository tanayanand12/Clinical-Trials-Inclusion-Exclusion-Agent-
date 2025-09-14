# doc_retriever.py
import logging
from typing import Dict, List, Any
import os

from enhanced_vectorization import VectorizationModule
from enhanced_faiss_db_manager import EnhancedFaissVectorDB
from gcp_storage_adapter import GCPStorageAdapter

logger = logging.getLogger(__name__)

class DocumentRetriever:
    """Document retrieval system for searching through document indexes."""
    
    def __init__(self):
        """Initialize document retriever."""
        self.vectorizer = VectorizationModule()
        self.docs_storage = GCPStorageAdapter(
            bucket_name=os.getenv("GCP_BUCKET", "intraintel-cloudrun-clinical-volume"),
            credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "service_account_credentials.json")
        )
    
    async def retrieve(
        self,
        query: str,
        model_id: str,
        top_k: int = 8,
        use_query_enrichment: bool = True
    ) -> Dict[str, Any]:
        """
        Perform document retrieval from document index with optional query enrichment.
        
        Args:
            query: User query
            model_id: Model ID for docs index
            top_k: Number of doc results to retrieve
            use_query_enrichment: Whether to use query enrichment for inclusion/exclusion criteria
            
        Returns:
            Document retrieval results
        """
        try:
            # Initial retrieval
            initial_results = await self._retrieve_docs(query, model_id, top_k)
            
            enriched_query = query
            final_results = initial_results
            
            # Apply query enrichment if enabled and we have initial results
            if use_query_enrichment and initial_results:
                enriched_query = self._build_enriched_query(query, initial_results)
                
                # Re-retrieve with enriched query if it's different
                if enriched_query != query:
                    logger.info("Re-retrieving documents with enriched inclusion/exclusion query")
                    final_results = await self._retrieve_docs(enriched_query, model_id, top_k)
            
            return {
                "original_query": query,
                "enriched_query": enriched_query,
                "results": final_results,
                "total_results": len(final_results),
                "query_was_enriched": enriched_query != query
            }
            
        except Exception as e:
            logger.error(f"Error in document retrieval: {e}")
            return {
                "original_query": query,
                "enriched_query": query,
                "results": [],
                "total_results": 0,
                "query_was_enriched": False,
                "error": str(e)
            }
    
    async def _retrieve_docs(self, query: str, model_id: str, top_k: int) -> List[Dict[str, Any]]:
        """Retrieve from docs index."""
        try:
            # Download docs index
            docs_index_path = f"gcp-indexes/{model_id}"
            if not self.docs_storage.download_index_using_model_id(model_id, docs_index_path):
                logger.error(f"Failed to download docs index: {model_id}")
                return []
            
            # Load docs vector database
            docs_db = EnhancedFaissVectorDB()
            if not docs_db.load(docs_index_path):
                logger.error(f"Failed to load docs index: {docs_index_path}")
                return []
            
            # Get query embedding with matching dimension
            docs_dim = docs_db.get_dimension()
            query_embedding = self.vectorizer.get_query_vector_auto_dim(query, docs_dim)
            
            # Search docs
            results, scores = docs_db.similarity_search(query_embedding, k=top_k)
            
            # Add scores to results
            for i, result in enumerate(results):
                result['similarity_score'] = float(scores[i]) if i < len(scores) else 0.0
                result['source_type'] = 'docs'
            
            logger.info(f"Retrieved {len(results)} document results")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def _build_enriched_query(self, original_query: str, docs_results: List[Dict[str, Any]]) -> str:
        """
        Build enriched query based on docs context for inclusion/exclusion criteria.
        
        Args:
            original_query: Original user query
            docs_results: Results from initial docs retrieval
            
        Returns:
            Enriched query string with inclusion/exclusion terms
        """
        if not docs_results:
            return original_query
        
        try:
            # Extract key inclusion/exclusion terms from top docs
            enrichment_terms = []
            
            for result in docs_results[:3]:  # Use top 3 docs for enrichment
                text = result.get('text', '')
                
                # Extract potential inclusion/exclusion criteria terms
                criteria_terms = self._extract_inclusion_exclusion_terms(text)
                enrichment_terms.extend(criteria_terms)
            
            # Remove duplicates and limit
            unique_terms = list(set(enrichment_terms))[:5]
            
            if unique_terms:
                enriched_query = f"{original_query} {' '.join(unique_terms)}"
                logger.info(f"Enriched query with inclusion/exclusion terms: {unique_terms}")
                return enriched_query
            
            return original_query
            
        except Exception as e:
            logger.error(f"Error building enriched query: {e}")
            return original_query
    
    def _extract_inclusion_exclusion_terms(self, text: str) -> List[str]:
        """
        Extract relevant inclusion/exclusion criteria terms from text.
        
        Args:
            text: Input text
            
        Returns:
            List of inclusion/exclusion criteria terms
        """
        # Inclusion/exclusion specific keywords
        inclusion_exclusion_keywords = [
            # Eligibility criteria
            'inclusion criteria', 'exclusion criteria', 'eligibility', 'eligible',
            'qualify', 'qualified', 'disqualified', 'ineligible',
            
            # Age-related terms
            'age', 'years old', 'adult', 'pediatric', 'elderly', 'geriatric',
            'minimum age', 'maximum age', 'age range',
            
            # Gender and demographics
            'gender', 'sex', 'male', 'female', 'ethnicity', 'race',
            'demographic', 'population',
            
            # Medical conditions
            'diagnosis', 'diagnosed', 'condition', 'disease', 'disorder',
            'medical history', 'comorbidity', 'comorbid',
            
            # Performance status
            'performance status', 'ECOG', 'Karnofsky', 'functional status',
            'ambulatory', 'bedridden',
            
            # Laboratory values
            'laboratory', 'lab values', 'blood work', 'CBC', 'chemistry panel',
            'liver function', 'kidney function', 'renal function',
            'creatinine', 'hemoglobin', 'platelet count',
            
            # Prior treatments
            'prior therapy', 'previous treatment', 'treatment naive',
            'chemotherapy', 'radiation', 'surgery', 'immunotherapy',
            
            # Consent and compliance
            'informed consent', 'consent', 'willing to participate',
            'compliance', 'adherence', 'follow-up',
            
            # Pregnancy and contraception
            'pregnancy', 'pregnant', 'breastfeeding', 'contraception',
            'reproductive potential', 'childbearing potential',
            
            # Organ function
            'organ function', 'cardiac function', 'pulmonary function',
            'hepatic function', 'adequate organ function'
        ]
        
        text_lower = text.lower()
        found_terms = []
        
        for keyword in inclusion_exclusion_keywords:
            if keyword in text_lower:
                found_terms.append(keyword)
        
        return found_terms[:4]  # Limit to top 4 terms for focused enrichment
    

#############################################
# old code
#############################################

# # doc_retriever.py
# import logging
# from typing import Dict, List, Any
# import os

# from src.enhanced_vectorization import VectorizationModule
# from src.enhanced_faiss_db_manager import EnhancedFaissVectorDB
# from src.gcp_storage_adapter import GCPStorageAdapter

# logger = logging.getLogger(__name__)

# class DocumentRetriever:
#     """Document retrieval system for searching through document indexes."""
    
#     def __init__(self):
#         """Initialize document retriever."""
#         self.vectorizer = VectorizationModule()
#         self.docs_storage = GCPStorageAdapter(
#             bucket_name=os.getenv("GCP_BUCKET", "intraintel-cloudrun-clinical-volume"),
#             credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "service_account_credentials.json")
#         )
    
#     async def retrieve(
#         self,
#         query: str,
#         model_id: str,
#         top_k: int = 8
#     ) -> Dict[str, Any]:
#         """
#         Perform document retrieval from document index.
        
#         Args:
#             query: User query
#             model_id: Model ID for docs index
#             top_k: Number of doc results to retrieve
            
#         Returns:
#             Document retrieval results
#         """
#         try:
#             results = await self._retrieve_docs(query, model_id, top_k)
            
#             return {
#                 "query": query,
#                 "results": results,
#                 "total_results": len(results)
#             }
            
#         except Exception as e:
#             logger.error(f"Error in document retrieval: {e}")
#             return {
#                 "query": query,
#                 "results": [],
#                 "total_results": 0,
#                 "error": str(e)
#             }
    
#     async def _retrieve_docs(self, query: str, model_id: str, top_k: int) -> List[Dict[str, Any]]:
#         """Retrieve from docs index."""
#         try:
#             # Download docs index
#             docs_index_path = f"gcp-indexes/{model_id}"
#             if not self.docs_storage.download_index_using_model_id(model_id, docs_index_path):
#                 logger.error(f"Failed to download docs index: {model_id}")
#                 return []
            
#             # Load docs vector database
#             docs_db = EnhancedFaissVectorDB()
#             if not docs_db.load(docs_index_path):
#                 logger.error(f"Failed to load docs index: {docs_index_path}")
#                 return []
            
#             # Get query embedding with matching dimension
#             docs_dim = docs_db.get_dimension()
#             query_embedding = self.vectorizer.get_query_vector_auto_dim(query, docs_dim)
            
#             # Search docs
#             results, scores = docs_db.similarity_search(query_embedding, k=top_k)
            
#             # Add scores to results
#             for i, result in enumerate(results):
#                 result['similarity_score'] = float(scores[i]) if i < len(scores) else 0.0
#                 result['source_type'] = 'docs'
            
#             logger.info(f"Retrieved {len(results)} document results")
#             return results
            
#         except Exception as e:
#             logger.error(f"Error retrieving documents: {e}")
#             return []