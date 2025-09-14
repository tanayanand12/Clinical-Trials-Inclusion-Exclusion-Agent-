from typing import Dict, Any, Optional
import os
import logging
import openai # type: ignore
from datetime import datetime
from agent_base import AgentBase

# Import your clinical trials pipeline
from clinical_trials_agent1.clinical_trials_rag_pipeline import ClinicalTrialsRAGPipeline

# Configure logging with UTF-8 encoding handling
logger = logging.getLogger(__name__)

# Set environment variable for better Windows console encoding
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

class ClinicalTrialsAgent(AgentBase):
    """
    Agent wrapper for clinical trials RAG system that answers queries 
    based on clinical trials data and research.
    """
    
    def __init__(self):
        """Initialize the Clinical Trials agent with its pipeline."""
        try:
            # Initialize OpenAI client
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found in environment variables")
                
            self.openai_client = openai.Client(api_key=api_key)
            
            # Initialize the clinical trials pipeline
            self.pipeline = ClinicalTrialsRAGPipeline(
                openai_client=self.openai_client,
                model_name="gpt-4-turbo",
                embedding_model="text-embedding-ada-002",
                max_trials=25,
                max_context_length=8000,
                chunk_size=1000,
                chunk_overlap=200
            )
            
            logger.info("Clinical Trials agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Clinical Trials agent: {str(e)}")
            raise
        
    def get_summary(self) -> str:
        """Return a summary of the agent's capabilities."""
        return "Clinical trials RAG system that answers queries based on clinical trials data and research studies"
        
    def query(self, question: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a query against the clinical trials knowledge base.
        
        Parameters
        ----------
        question : str
            The user's question
        context : Dict[str, Any], optional
            Additional context parameters:
            - top_k : int, number of results to retrieve (default: 10)
            - max_trials : int, maximum trials to process (default: 25)
            
        Returns
        -------
        Dict[str, Any]
            Response dictionary containing answer, citations, and confidence score
        """
        try:
            # Extract context parameters
            context = context or {}
            top_k = context.get('top_k', 10)
            max_trials = context.get('max_trials', 25)
            
            logger.info(f"Processing clinical trials query with top_k={top_k}, max_trials={max_trials}")
            
            # Update pipeline parameters if different from defaults
            if max_trials != 25:
                self.pipeline.max_trials = max_trials
            
            start_time = datetime.now()
            
            # Process the query using the clinical trials pipeline
            result = self.pipeline.process_query(
                query=question,
                top_k=top_k
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Clinical trials query processed in {processing_time:.2f} seconds")
            
            # Check if there was an error in the result
            if "error" in result:
                raise ValueError(result["error"])
            
            # Extract and format the response
            answer = result.get("answer", "")
            citations = result.get("citations", [])
            
            # If citations is not a list, try to convert or create empty list
            if not isinstance(citations, list):
                if isinstance(citations, str):
                    # If citations is a string, split it or wrap it in a list
                    citations = [citations] if citations else []
                else:
                    citations = []
            
            # Calculate confidence based on result quality
            confidence = self._calculate_confidence(result, processing_time)
            
            return {
                "answer": answer,
                "citations": citations,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Error in Clinical Trials agent: {str(e)}", exc_info=True)
            return {
                "answer": f"Error processing clinical trials query: {str(e)}",
                "citations": [],
                "confidence": 0.0
            }
    
    def _calculate_confidence(self, result: Dict[str, Any], processing_time: float) -> float:
        """
        Calculate confidence score based on result quality and processing metrics.
        
        Parameters
        ----------
        result : Dict[str, Any]
            The result from the clinical trials pipeline
        processing_time : float
            Time taken to process the query
            
        Returns
        -------
        float
            Confidence score between 0.0 and 1.0
        """
        try:
            base_confidence = 0.8  # Base confidence for clinical trials agent
            
            # Adjust based on answer length (longer answers might be more comprehensive)
            answer_length = len(result.get("answer", ""))
            if answer_length > 500:
                base_confidence += 0.05
            elif answer_length < 100:
                base_confidence -= 0.1
            
            # Adjust based on number of citations
            citations = result.get("citations", [])
            if isinstance(citations, list) and len(citations) > 3:
                base_confidence += 0.1
            elif isinstance(citations, list) and len(citations) == 0:
                base_confidence -= 0.15
            
            # Adjust based on processing time (very fast might indicate cached/simple response)
            if processing_time < 2.0:
                base_confidence -= 0.05
            elif processing_time > 10.0:
                base_confidence -= 0.1
            
            # Ensure confidence is within bounds
            return max(0.0, min(1.0, base_confidence))
            
        except Exception as e:
            logger.warning(f"Error calculating confidence: {str(e)}")
            return 0.75  # Default confidence on error