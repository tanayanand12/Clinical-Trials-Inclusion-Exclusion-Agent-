# clinical_trials_rag_pipeline.py
import logging
import os
import openai # type: ignore
import time
from typing import Dict, Any, Optional, List
from .fetcher import ClinicalTrialsFetcherAgent
from .clinical_trials_chunker import ClinicalTrialsChunker
from .clinical_trials_vectorizer import ClinicalTrialsVectorizer
from .clinical_trials_context_extractor import ClinicalTrialsContextExtractor
from .clinical_trials_rag_module import ClinicalTrialsRAGModule
# from .endpoint_prediction_integration import EndpointPredictionAPIIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ClinicalTrialsRAGPipeline:
    """
    End-to-end RAG pipeline for clinical trials data.
    Fetches, chunks, vectorizes, and retrieves relevant clinical trial information to answer user queries.
    """
    
    def __init__(self, 
                 openai_client=None or  openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
                 model_name: str = os.getenv("MODEL_ID_GPT5") or "gpt-4o",
                 embedding_model: str = "text-embedding-ada-002",
                 max_trials: int = 20,
                 max_chunks_per_trial: int = 10,
                 max_context_length: int = 80000,
                 chunk_size: int = 10000,
                 chunk_overlap: int = 500):
        """
        Initialize the Clinical Trials RAG Pipeline.
        
        Args:
            openai_client: OpenAI client instance (optional)
            model_name: OpenAI model for answer generation
            embedding_model: OpenAI model for embeddings
            max_trials: Maximum number of trials to fetch
            max_chunks_per_trial: Maximum chunks to create per trial
            max_context_length: Maximum context length for RAG
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.max_trials = max_trials
        self.max_chunks_per_trial = max_chunks_per_trial
        
        # Initialize components
        logger.info("Initializing Clinical Trials RAG Pipeline components...")
        
        try:
            # Initialize fetcher
            # self.fetcher = ClinicalTrialsFetcherAgent(
            #     openai_client=openai_client,
            #     model=model_name
            # )


            # In clinical_trials_rag_pipeline.py, line ~77
            self.fetcher = ClinicalTrialsFetcherAgent(
                openai_client=openai_client,
                model=model_name
            )
            if not openai_client:
                logger.warning("OpenAI client not provided to fetcher. URL generation will fail.")

                
            logger.info("[OK] ClinicalTrialsFetcherAgent initialized")
            
            # Initialize chunker
            self.chunker = ClinicalTrialsChunker(
                max_chunk_size=chunk_size,
                overlap_size=chunk_overlap
            )
            logger.info("[OK] ClinicalTrialsChunker initialized")
            
            # Initialize vectorizer
            self.vectorizer = ClinicalTrialsVectorizer(
                openai_model=embedding_model
            )
            logger.info("[OK] ClinicalTrialsVectorizer initialized")
            
            # Initialize context extractor
            self.context_extractor = ClinicalTrialsContextExtractor(
                max_context_length=max_context_length
            )
            logger.info("[OK] ClinicalTrialsContextExtractor initialized")
            
            # Initialize RAG module
            self.rag_module = ClinicalTrialsRAGModule(
                model_name=model_name
            )
            logger.info("[OK] ClinicalTrialsRAGModule initialized")
            
            # self.endpoint_predictor = EndpointPredictionAPIIntegration()
            # logger.info("[OK] EndpointPredictionAPIIntegration initialized")

            logger.info("Clinical Trials RAG Pipeline initialization complete!")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    def fetch_clinical_trials_data(self, query: str) -> Dict[str, Any]:
        """
        Fetch clinical trials data based on user query.
        
        Args:
            query: User query about clinical trials
            
        Returns:
            Dictionary containing fetched data and metadata
        """
        logger.info(f"Fetching clinical trials data for query: '{query}'")
        
        try:
            # Use the fetcher to analyze query and fetch relevant trials
            result = self.fetcher.analyze_user_query(query)
            
            if not result.get('success', False):
                logger.error(f"Failed to fetch clinical trials data: {result.get('error', 'Unknown error')}")
                return {
                    'success': False,
                    'error': result.get('error', 'Failed to fetch clinical trials data'),
                    'data': None
                }
            else: 
                logger.info(f'Fetched {result.get("studies_returned", 0)} studies successfully')
            
            trials_data = result.get('data', {})
            total_count = result.get('total_count', 0)
            
            logger.info(f"Successfully fetched {total_count} clinical trials")
            
            return {
                'success': True,
                'data': trials_data,
                'total_count': total_count,
                'studies_returned': result.get('studies_returned', 0),
                'source_url': result.get('source_url', ''),
                'query_analysis': result.get('query_analysis', {})
            }
            
        except Exception as e:
            logger.error(f"Error fetching clinical trials data: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': None
            }
    
    def process_and_chunk_data(self, trials_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process and chunk the fetched clinical trials data.
        
        Args:
            trials_data: Raw clinical trials data from API
            
        Returns:
            List of chunks ready for vectorization
        """
        logger.info("Processing and chunking clinical trials data...")
        
        try:
            # Chunk the clinical trials data
            chunks = self.chunker.chunk_clinical_trials_data(trials_data)
            
            # Limit chunks if too many
            if len(chunks) > self.max_trials * self.max_chunks_per_trial:
                chunks = chunks[:self.max_trials * self.max_chunks_per_trial]
                logger.info(f"Limited chunks to {len(chunks)} for processing efficiency")
            
            logger.info(f"Created {len(chunks)} chunks from clinical trials data")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing and chunking data: {e}")
            return []
    
    def vectorize_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Vectorize the chunks using OpenAI embeddings.
        
        Args:
            chunks: List of chunks to vectorize
            
        Returns:
            Dictionary mapping chunk IDs to embeddings and metadata
        """
        logger.info(f"Vectorizing {len(chunks)} chunks...")
        
        try:
            # Embed all chunks
            chunk_embeddings = self.vectorizer.embed_chunks(chunks)
            
            logger.info(f"Successfully vectorized {len(chunk_embeddings)} chunks")
            return chunk_embeddings
            
        except Exception as e:
            logger.error(f"Error vectorizing chunks: {e}")
            return {}
    
    def retrieve_relevant_context(self, 
                                 query: str, 
                                 chunk_embeddings: Dict[str, Any],
                                 top_k: int = 10) -> Dict[str, Any]:
        """
        Retrieve relevant context for the query.
        
        Args:
            query: User query
            chunk_embeddings: Vectorized chunks
            top_k: Number of top chunks to retrieve
            
        Returns:
            Dictionary containing context and metadata
        """
        logger.info(f"Retrieving relevant context for query: '{query}'")
        
        try:
            # Embed the query
            query_embedding = self.vectorizer.embed_query(query)
            
            # Extract relevant context
            context_result = self.context_extractor.extract_context(
                query=query,
                query_embedding=query_embedding,
                chunk_embeddings=chunk_embeddings,
                vectorizer=self.vectorizer,
                top_k=top_k
            )
            
            logger.info(f"Retrieved context from {context_result.get('chunk_count', 0)} relevant chunks")
            return context_result
            
        except Exception as e:
            logger.error(f"Error retrieving relevant context: {e}")
            return {
                'context': f'Error retrieving relevant context: {str(e)}',
                'studies': [],
                'chunk_count': 0
            }
    
    def generate_final_answer(self, 
                             query: str, 
                             context_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate the final answer using RAG.
        
        Args:
            query: User query
            context_result: Context and metadata from retrieval
            
        Returns:
            Dictionary containing the final answer and metadata
        """
        logger.info("Generating final answer using RAG...")
        
        try:
            # Generate answer using RAG module
            answer_result = self.rag_module.generate_answer(
                query=query,
                context=context_result.get('context', ''),
                studies=context_result.get('studies', [])
            )
            
            # Validate response quality
            validated_result = self.rag_module.validate_response_quality(answer_result, query)
            
            logger.info(f"Generated answer with quality score: {validated_result.get('quality_assessment', {}).get('overall_score', 0):.2f}")
            return validated_result
            
        except Exception as e:
            logger.error(f"Error generating final answer: {e}")
            return {
                'answer': f'I encountered an error while generating the answer: {str(e)}',
                'citations': [],
                'studies': context_result.get('studies', []),
                'metadata': {'error': str(e)}
            }
    
    def process_query(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Main method to process a clinical trials query end-to-end.
        
        Args:
            query: User query about clinical trials
            top_k: Number of top relevant chunks to use for context
            
        Returns:
            Dictionary containing the complete response
        """
        start_time = time.time()
        logger.info(f"Starting end-to-end processing for query: '{query}'")
        
        try:
            # Step 1: Fetch clinical trials data
            fetch_result = self.fetch_clinical_trials_data(query)
            
            if not fetch_result.get('success', False):
                return {
                    'query': query,
                    'success': False,
                    'error': fetch_result.get('error', 'Failed to fetch clinical trials data'),
                    'processing_time': time.time() - start_time
                }
            
            trials_data = fetch_result['data']
            
            # Step 2: Process and chunk data
            chunks = self.process_and_chunk_data(trials_data)
            
            if not chunks:
                return {
                    'query': query,
                    'success': False,
                    'error': 'No processable chunks created from clinical trials data',
                    'processing_time': time.time() - start_time
                }
            
            # Step 3: Vectorize chunks
            chunk_embeddings = self.vectorize_chunks(chunks)
            
            if not chunk_embeddings:
                return {
                    'query': query,
                    'success': False,
                    'error': 'Failed to vectorize clinical trials chunks',
                    'processing_time': time.time() - start_time
                }
            
            # Step 4: Retrieve relevant context
            context_result = self.retrieve_relevant_context(query, chunk_embeddings, top_k)
            
            # Step 5: Generate final answer
            answer_result = self.generate_final_answer(query, context_result)

            # Step 6: Endpoint prediction integration
            # endpoint_results = self.endpoint_predictor.process_query(query) #implement this part in async way later
            # endpoint_results = str(endpoint_results) if endpoint_results else "No endpoint prediction available"
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Compile final result
            final_result = {
                'query': query,
                'success': True,
                'answer': answer_result.get('answer', ''), # + f"\n\nEndpoint Prediction: {endpoint_results}",
                'citations': answer_result.get('citations', []),
                'studies': answer_result.get('studies', []),
                'metadata': {
                    **answer_result.get('metadata', {}),
                    'processing_time': processing_time,
                    'total_trials_fetched': fetch_result.get('total_count', 0),
                    'trials_processed': fetch_result.get('studies_returned', 0),
                    'chunks_created': len(chunks),
                    'chunks_vectorized': len(chunk_embeddings),
                    'relevant_chunks': context_result.get('chunk_count', 0),
                    'fetch_metadata': {
                        'source_url': fetch_result.get('source_url', ''),
                        'query_analysis': fetch_result.get('query_analysis', {})
                    }
                },
                # 'endpoint_prediction': endpoint_results,
                'quality_assessment': answer_result.get('quality_assessment', {})
            }
            
            logger.info(f"Successfully processed query in {processing_time:.2f} seconds")
            return final_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error in end-to-end processing: {e}")
            
            return {
                'query': query,
                'success': False,
                'error': str(e),
                'processing_time': processing_time,
                'metadata': {}
            }
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get the current status of the pipeline components.
        
        Returns:
            Dictionary containing pipeline status information
        """
        return {
            'pipeline_initialized': True,
            'components': {
                'fetcher': self.fetcher is not None,
                'chunker': self.chunker is not None,
                'vectorizer': self.vectorizer is not None,
                'context_extractor': self.context_extractor is not None,
                'rag_module': self.rag_module is not None
            },
            'configuration': {
                'max_trials': self.max_trials,
                'max_chunks_per_trial': self.max_chunks_per_trial,
                'chunker_max_size': self.chunker.max_chunk_size,
                'chunker_overlap': self.chunker.overlap_size,
                'vectorizer_model': self.vectorizer.openai_model,
                'context_max_length': self.context_extractor.max_context_length,
                'rag_model': self.rag_module.model_name
            }
        }


# Example usage and demo function
def demo_clinical_trials_rag():
    """
    Demonstrate the Clinical Trials RAG Pipeline.
    """
    try:
        # Initialize pipeline
        pipeline = ClinicalTrialsRAGPipeline()
        
        # Example queries
        queries = [
            'How many recruiting clinical trials are there for Type 2 Diabetes in India?',
            'What are the most common interventions for diabetes studies?',
            'Find completed trials for cardiovascular disease treatments.'
        ]
        
        for query in queries:
            print(f"\n{'='*50}")
            print(f"Query: {query}")
            print('='*50)
            
            result = pipeline.process_query(query)
            
            if result['success']:
                print(f"Answer: {result['answer'][:500]}...")
                print(f"\nStudies found: {len(result['studies'])}")
                print(f"Processing time: {result['metadata']['processing_time']:.2f} seconds")
                print(f"Quality score: {result['quality_assessment']['overall_score']:.2f}")
            else:
                print(f"Error: {result['error']}")
        
    except Exception as e:
        print(f"Demo failed: {e}")


if __name__ == "__main__":
    demo_clinical_trials_rag()