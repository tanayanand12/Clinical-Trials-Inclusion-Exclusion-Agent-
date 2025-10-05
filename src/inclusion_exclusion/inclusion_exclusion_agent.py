from typing import Dict, Any, Optional
import os
import logging
import openai # type: ignore
from datetime import datetime
from dotenv import load_dotenv # type: ignore
from .agent_base import AgentBase

# Import the clinical trials pipeline
from src.clinical_trials.clinical_trials_agent_wrapper import ClinicalTrialsRAGPipeline

# Configure logging with UTF-8 encoding handling
logger = logging.getLogger(__name__)

# Set environment variable for better Windows console encoding
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

class ClinicalTrialInclusionExclusionAgent(AgentBase):
    """
    Agent wrapper for clinical trials inclusion/exclusion criteria analysis system 
    that answers queries based on clinical trial eligibility requirements and 
    demographic factors.
    """
    
    def __init__(self):
        """Initialize the Clinical Trial Inclusion/Exclusion agent with its pipeline."""
        try:
            # Load environment variables
            load_dotenv()
            
            # Initialize OpenAI client
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found in environment variables")
                
            self.openai_client = openai.OpenAI(api_key=api_key)
            
            # Initialize the clinical trials pipeline
            self.pipeline = ClinicalTrialsRAGPipeline(openai_client=self.openai_client)
            
            logger.info("Clinical Trial Inclusion/Exclusion agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Clinical Trial Inclusion/Exclusion agent: {str(e)}")
            raise
        
    def get_summary(self) -> str:
        """Return a summary of the agent's capabilities."""
        return ("Clinical trial inclusion/exclusion criteria analysis system that answers queries "
                "about eligibility requirements, demographic factors, and patient selection criteria "
                "in clinical trials and research studies")
        
    def query(self, question: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a query about clinical trial inclusion/exclusion criteria.
        
        Parameters
        ----------
        question : str
            The user's question about inclusion/exclusion criteria
        context : Dict[str, Any], optional
            Additional context parameters:
            - top_k : int, number of results to retrieve (default: 10)
            
        Returns
        -------
        Dict[str, Any]
            Response dictionary containing answer, citations, and confidence score
        """
        try:
            # Extract context parameters
            context = context or {}
            
            logger.info("Processing clinical trial inclusion/exclusion criteria query")
            
            start_time = datetime.now()
            
            # Process the query using the clinical trials pipeline
            result = self.pipeline.process_query(question)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Clinical trial inclusion/exclusion query processed in {processing_time:.2f} seconds")
            
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
            logger.error(f"Error in Clinical Trial Inclusion/Exclusion agent: {str(e)}", exc_info=True)
            return {
                "answer": f"Error processing clinical trial inclusion/exclusion query: {str(e)}",
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
            base_confidence = 0.85  # Base confidence for inclusion/exclusion criteria agent
            
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
            return 0.8  # Default confidence on error
    
    def analyze_inclusion_exclusion_criteria(self, trial_data: str) -> Dict[str, Any]:
        """
        Specialized method to analyze inclusion/exclusion criteria from trial data.
        
        Parameters
        ----------
        trial_data : str
            Clinical trial data or description
            
        Returns
        -------
        Dict[str, Any]
            Analysis of inclusion/exclusion criteria
        """
        query = f"Analyze the inclusion and exclusion criteria in this clinical trial data: {trial_data}"
        return self.query(query)
    
    def get_demographic_factors(self, question: str) -> Dict[str, Any]:
        """
        Specialized method to query demographic factors used in clinical trials.
        
        Parameters
        ----------
        question : str
            Question about demographic factors
            
        Returns
        -------
        Dict[str, Any]
            Information about demographic factors in clinical trials
        """
        demographic_query = f"What demographic factors (age, gender, ethnicity, etc.) are relevant to: {question}"
        return self.query(demographic_query)
