# comprehensive_rag_pipeline_agent.py
import logging
import json
from typing import Dict, List, Any, Optional
import os
import asyncio
from datetime import datetime
import openai # type: ignore
from dotenv import load_dotenv # type: ignore
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from dotenv import load_dotenv # type: ignore
load_dotenv()

# Import the components
from src.retrieval.doc_retriever import DocumentRetriever
from src.inclusion_exclusion.inclusion_exclusion_agent import ClinicalTrialInclusionExclusionAgent

# Configure logging
logger = logging.getLogger(__name__)

class ComprehensiveRAGPipelineAgent:
    """
    Comprehensive RAG Data Pipeline Agent that orchestrates:
    1. Local data retrieval and query enhancement
    2. Inclusion/exclusion criteria analysis
    3. Final response generation using OpenAI
    """
    
    def __init__(
        self,
        model_name: str = os.getenv("MODEL_ID_GPT5", "gpt-5-2025-08-07"),
        max_context_length: int = 8000,
        local_data_top_k: int = 8,
        criteria_top_k: int = 10
    ):
        """
        Initialize the comprehensive RAG pipeline agent.
        
        Args:
            model_name: OpenAI model to use for final response generation
            max_context_length: Maximum context length for OpenAI model
            local_data_top_k: Number of local data chunks to retrieve
            criteria_top_k: Number of inclusion/exclusion results to retrieve
        """
        try:
            # Load environment variables
            load_dotenv()
            
            # Initialize OpenAI client
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found in environment variables")
                
            self.openai_client = openai.OpenAI(api_key=api_key)
            self.model_name = model_name
            self.max_context_length = max_context_length
            self.local_data_top_k = local_data_top_k
            self.criteria_top_k = criteria_top_k
            
            # Initialize components
            self.document_retriever = DocumentRetriever()
            self.criteria_agent = ClinicalTrialInclusionExclusionAgent()
            
            logger.info("Comprehensive RAG Pipeline Agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Comprehensive RAG Pipeline Agent: {str(e)}")
            raise
    
    async def process_comprehensive_query(
        self,
        user_query: str,
        local_model_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a comprehensive query through the full RAG pipeline.
        
        Args:
            user_query: Original user query
            local_model_id: Model ID for local data retrieval
            context: Additional context parameters
            
        Returns:
            Comprehensive response with all pipeline results
        """
        start_time = datetime.now()
        pipeline_results = {
            "user_query": user_query,
            "local_model_id": local_model_id,
            "pipeline_steps": {},
            "final_response": "",
            "processing_time": 0.0,
            "error": None
        }
        
        try:
            logger.info(f"Starting comprehensive RAG pipeline for query: {user_query[:100]}...")
            
            # Step 1: Retrieve local data and enhance query
            logger.info("Step 1: Retrieving local data and enhancing query")
            local_data_results = await self._retrieve_local_data_and_enhance_query(
                user_query, local_model_id
            )
            pipeline_results["pipeline_steps"]["local_data_retrieval"] = local_data_results
            
            # Step 2: Use enhanced query for inclusion/exclusion criteria analysis
            logger.info("Step 2: Analyzing inclusion/exclusion criteria")
            enhanced_query = local_data_results.get("enriched_query", user_query)
            criteria_results = await self._analyze_inclusion_exclusion_criteria(enhanced_query)
            pipeline_results["pipeline_steps"]["criteria_analysis"] = criteria_results
            
            # Step 3: Combine all results for comprehensive response
            logger.info("Step 3: Generating comprehensive response")
            final_response = await self._generate_comprehensive_response(
                user_query, local_data_results, criteria_results
            )
            pipeline_results["final_response"] = final_response
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            pipeline_results["processing_time"] = processing_time
            
            logger.info(f"Comprehensive RAG pipeline completed in {processing_time:.2f} seconds")
            
            return pipeline_results
            
        except Exception as e:
            error_msg = f"Error in comprehensive RAG pipeline: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            pipeline_results["error"] = error_msg
            pipeline_results["processing_time"] = (datetime.now() - start_time).total_seconds()
            
            return pipeline_results
    
    async def _retrieve_local_data_and_enhance_query(
        self, 
        user_query: str, 
        model_id: str
    ) -> Dict[str, Any]:
        """
        Step 1: Retrieve local data and enhance the user query.
        
        Args:
            user_query: Original user query
            model_id: Model ID for local data
            
        Returns:
            Local data retrieval results with enhanced query
        """
        try:
            # Use document retriever with query enrichment enabled
            results = await self.document_retriever.retrieve(
                query=user_query,
                model_id=model_id,
                top_k=self.local_data_top_k,
                use_query_enrichment=True
            )
            
            logger.info(f"Retrieved {results.get('total_results', 0)} local data chunks")
            logger.info(f"Query enriched: {results.get('query_was_enriched', False)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in local data retrieval: {str(e)}")
            return {
                "original_query": user_query,
                "enriched_query": user_query,
                "results": [],
                "total_results": 0,
                "query_was_enriched": False,
                "error": str(e)
            }
    
    async def _analyze_inclusion_exclusion_criteria(self, enhanced_query: str) -> Dict[str, Any]:
        """
        Step 2: Analyze inclusion/exclusion criteria using enhanced query.
        
        Args:
            enhanced_query: Enhanced query from local data retrieval
            
        Returns:
            Inclusion/exclusion criteria analysis results
        """
        try:
            # Use inclusion/exclusion agent to analyze criteria
            criteria_results = self.criteria_agent.query(
                question=enhanced_query,
                context={"top_k": self.criteria_top_k}
            )
            
            logger.info(f"Inclusion/exclusion analysis completed with confidence: {criteria_results.get('confidence', 0.0)}")
            
            return criteria_results
            
        except Exception as e:
            logger.error(f"Error in inclusion/exclusion criteria analysis: {str(e)}")
            return {
                "answer": f"Error in criteria analysis: {str(e)}",
                "citations": [],
                "confidence": 0.0,
                "error": str(e)
            }
    
    async def _generate_comprehensive_response(
        self,
        original_query: str,
        local_data_results: Dict[str, Any],
        criteria_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Step 3: Generate comprehensive response using OpenAI.
        
        Args:
            original_query: Original user query
            local_data_results: Results from local data retrieval
            criteria_results: Results from criteria analysis
            
        Returns:
            Final comprehensive response
        """
        try:
            # Build context from all sources
            context_parts = []
            
            # Add local data context
            local_results = local_data_results.get("results", [])
            if local_results:
                context_parts.append("=== LOCAL CLINICAL TRIALS DATA ===")
                for i, result in enumerate(local_results[:5], 1):  # Limit to top 5 for context
                    text = result.get("text", "")[:500]  # Limit text length
                    score = result.get("similarity_score", 0.0)
                    context_parts.append(f"Local Data {i} (Score: {score:.3f}):\n{text}\n")
            
            # Add inclusion/exclusion criteria context
            criteria_answer = criteria_results.get("answer", "")
            if criteria_answer:
                context_parts.append("=== INCLUSION/EXCLUSION CRITERIA ANALYSIS ===")
                context_parts.append(criteria_answer)
            
            # Add citations
            all_citations = []
            all_citations.extend(criteria_results.get("citations", []))
            
            if all_citations:
                context_parts.append("=== CITATIONS ===")
                for i, citation in enumerate(all_citations[:10], 1):  # Limit citations
                    context_parts.append(f"{i}. {citation}")
            
            # Combine context
            full_context = "\n\n".join(context_parts)
            
            # Truncate context if too long
            if len(full_context) > self.max_context_length:
                full_context = full_context[:self.max_context_length] + "...[truncated]"
            
            # Create comprehensive prompt
            system_prompt = self._get_system_prompt()
            user_prompt = self._build_user_prompt(original_query, full_context, local_data_results, criteria_results)
            
            # Generate response using OpenAI
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                # temperature=0.3,
                # max_tokens=1500
                reasoning_effort="minimal",
                response_format={"type": "json_object"}  # <- enforce JSON
            )
            
            raw = response.choices[0].message.content or "{}"

            # Be robust: keep the original string AND parsed JSON
            try:
                payload = json.loads(raw)
            except Exception:
                payload = {"Inclusion": "- Not reported in cited records",
                        "Exclusion": "- Not reported in cited records",
                        "Similarity": []}

            return {
                # If your caller prints this, it'll show plain JSON text (no markdown)
                "comprehensive_answer": raw,
                # Use this in code for structured access
                "comprehensive_answer_parsed": payload,
                "context_used": {
                    "local_data_chunks": len(local_results),
                    "criteria_confidence": criteria_results.get("confidence", 0.0),
                    "total_citations": len(all_citations)
                },
                "model_used": self.model_name,
                "tokens_used": response.usage.total_tokens if response.usage else 0
            }
            
        except Exception as e:
            logger.error(f"Error generating comprehensive response: {str(e)}")
            return {
                "comprehensive_answer": f"Error generating final response: {str(e)}",
                "context_used": {},
                "error": str(e)
            }
    
#     def _get_system_prompt(self) -> str:
#         """Get the system prompt for OpenAI."""
#         return """You are a comprehensive clinical trials analysis assistant. You help researchers and clinicians by providing detailed, accurate information about clinical trials, with a special focus on inclusion and exclusion criteria.

# Your responses should be:
# 1. Comprehensive and well-structured
# 2. Based on the provided local data and external criteria analysis
# 3. Focused on answering the user's specific question
# 4. Include relevant details about inclusion/exclusion criteria when applicable
# 5. Cite sources when available
# 6. Acknowledge limitations or uncertainties

# Format your response with clear sections and bullet points when appropriate. Always prioritize accuracy and clinical relevance."""
    
    def _get_system_prompt(self) -> str:
        return (
            "You are a clinical trials inclusion/exclusion extraction assistant. "
            "You MUST output a single JSON object in the exact schema requested. "
            "Do not add prose, headings, code fences, or extra keys. "
            "Use the provided context (including the unmodified CITATIONS block) to extract NCT IDs and factual details. "
            "Your Inclusion/Exclusion bullets must be thorough and evidence-based."
        )


#     def _build_user_prompt(
#         self,
#         original_query: str,
#         context: str,
#         local_data_results: Dict[str, Any],
#         criteria_results: Dict[str, Any]
#     ) -> str:
#         """Build the user prompt for OpenAI."""
        
#         query_enhancement_info = ""
#         if local_data_results.get("query_was_enriched", False):
#             query_enhancement_info = f"""
# Original Query: {local_data_results.get('original_query', '')}
# Enhanced Query: {local_data_results.get('enriched_query', '')}
# """
        
#         return f"""Please provide a comprehensive answer to the following clinical trials question using all the available information:

# USER QUESTION: {original_query}
# {query_enhancement_info}
# AVAILABLE CONTEXT AND DATA:
# {context}

# Please provide a detailed, well-structured response that directly addresses the user's question. Include specific information about inclusion/exclusion criteria if relevant to the query. Structure your response with clear sections and cite the provided information where appropriate."""

    def _build_user_prompt(
        self,
        original_query: str,
        context: str,
        local_data_results: Dict[str, Any],
        criteria_results: Dict[str, Any]
    ) -> str:
        query_enhancement_info = ""
        if local_data_results.get("query_was_enriched", False):
            query_enhancement_info = (
                f"Original Query: {local_data_results.get('original_query', '')}\n"
                f"Enhanced Query: {local_data_results.get('enriched_query', '')}\n"
            )

        return (
            "USER QUESTION:\n"
            f"{original_query}\n\n"
            f"{query_enhancement_info}"
            "AVAILABLE CONTEXT AND DATA (READ-ONLY):\n"
            f"{context}\n\n"
            "IMPORTANT:\n"
            "- The block titled '=== CITATIONS ===' above contains the original citation text. Do NOT rewrite or alter it.\n"
            "- Use it only to extract valid NCT IDs (format: ^NCT[0-9]{8}$).\n\n"
            "OUTPUT FORMAT (STRICT JSON ONLY — NO PROSE, NO HEADINGS, NO BACKTICKS):\n"
            "{\n"
            '  "Inclusion": "<markdown bullets>",\n'
            '  "Exclusion": "<markdown bullets>",\n'
            '  "Similarity": [ { "CT_ID": "NCT########", "percentage_similarity": 0.0 } ]\n'
            "}\n\n"
            "CONTENT REQUIREMENTS FOR Inclusion/Exclusion (MAKE THEM DEEP & EXPLAINED):\n"
            "- Produce concise top-level bullets. For EACH bullet, add indented sub-bullets with:\n"
            "  • Rationale: why this criterion is included/excluded based on trial practice\n"
            "  • Typical thresholds/examples: concrete values (e.g., SBP ≥160 mmHg) or protocol patterns if present\n"
            "  • Edge cases/nuance: when allowed vs disallowed if such nuance appears in context\n"
            "  • Evidence: space-separated NCT IDs in square brackets (e.g., [NCT01234567 NCT07654321])\n"
            "- Aim for 3–8 top-level bullets per section when content exists. If a section lacks data, output a single bullet: \"- Not reported in cited records\".\n"
            "- Derive EVERY statement from the provided context and/or the CITATIONS block; do not invent facts or NCT IDs.\n"
            "- Inline citation rule: use square brackets with space-separated NCT IDs, e.g., [NCT01234567 NCT07654321].\n\n"
            "SIMILARITY ARRAY (FABRICATE SCORES CONSISTENTLY, INCLUDE ALL NCTs YOU USED):\n"
            "- Include EVERY distinct NCT ID you used in Inclusion/Exclusion (deduplicated). You may also include additional NCT IDs present in CONTEXT/CITATIONS if clearly relevant.\n"
            "- For each CT_ID, compute percentage_similarity on 0–100 with 1 decimal using:\n"
            "    IM = InterventionMatch (0–10), EEF = Endpoint/Eligibility Fit (0–10), DC = Design Comparability (0–10)\n"
            "    Similarity% = 100 * (0.4*(IM/10) + 0.4*(EEF/10) + 0.2*(DC/10))\n"
            "- Scoring guidance: default to conservative mids (IM=6.0, EEF=6.0, DC=5.0) when details are partial; penalize obvious mismatches; raise subscores only when context explicitly supports strong alignment.\n"
            "- Round to 1 decimal; numbers must be numeric (no % sign). Sort by descending percentage_similarity.\n\n"
            "STRICT RULES:\n"
            "- Return ONLY the JSON object. No extra keys or narrative.\n"
            "- Inclusion and Exclusion must be markdown bullets with the sub-bullets described above.\n"
            "- Validate NCT ID format before using.\n"
        )


    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline configuration and capabilities."""
        return {
            "pipeline_name": "Comprehensive RAG Data Pipeline Agent",
            "capabilities": [
                "Local clinical trials data retrieval and query enhancement",
                "Inclusion/exclusion criteria analysis",
                "Comprehensive response generation using OpenAI"
            ],
            "configuration": {
                "model_name": self.model_name,
                "max_context_length": self.max_context_length,
                "local_data_top_k": self.local_data_top_k,
                "criteria_top_k": self.criteria_top_k
            },
            "components": [
                "DocumentRetriever",
                "ClinicalTrialInclusionExclusionAgent",
                "OpenAI Integration"
            ]
        }

# Example usage and utility functions
async def main_example():
    """Example usage of the Comprehensive RAG Pipeline Agent."""
    
    # Initialize the pipeline agent
    pipeline_agent = ComprehensiveRAGPipelineAgent(
        model_name="gpt-4-turbo",
        max_context_length=8000,
        local_data_top_k=8,
        criteria_top_k=10
    )
    
    # Example queries
    example_queries = [
        "What are the age requirements for cancer clinical trials?",
        "Were there any demographic factors used for inclusion/exclusion, such as age, gender, or ethnicity?",
        "What are the common exclusion criteria for cardiovascular disease trials?",
        "How do inclusion criteria vary between Phase I and Phase III trials?"
    ]
    
    # Process each query
    for query in example_queries:
        print(f"\n{'='*60}")
        print(f"Processing Query: {query}")
        print(f"{'='*60}")
        
        try:
            # Process the comprehensive query
            result = await pipeline_agent.process_comprehensive_query(
                user_query=query,
                local_model_id="ct_epa_1"  # Replace with actual model ID
            )
            
            # Display results
            print(f"Processing Time: {result['processing_time']:.2f} seconds")
            print(f"Query Enhanced: {result['pipeline_steps']['local_data_retrieval'].get('query_was_enriched', False)}")
            print(f"Local Data Chunks: {result['pipeline_steps']['local_data_retrieval'].get('total_results', 0)}")
            print(f"Criteria Confidence: {result['pipeline_steps']['criteria_analysis'].get('confidence', 0.0):.2f}")
            
            if result.get('error'):
                print(f"Error: {result['error']}")
            else:
                print(f"\nFinal Response:\n{result['final_response']['comprehensive_answer']}")
                
        except Exception as e:
            print(f"Error processing query: {str(e)}")

class RAGPipelineManager:
    """
    Manager class for handling multiple RAG pipeline instances and batch processing.
    """
    
    def __init__(self):
        self.pipeline_agent = None
        self.results_history = []
    
    def initialize_pipeline(self, **kwargs):
        """Initialize the RAG pipeline with custom configuration."""
        self.pipeline_agent = ComprehensiveRAGPipelineAgent(**kwargs)
    
    async def batch_process_queries(
        self,
        queries: List[Dict[str, str]],
        max_concurrent: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batches.
        
        Args:
            queries: List of query dictionaries with 'query' and 'model_id' keys
            max_concurrent: Maximum concurrent processing
            
        Returns:
            List of results for all queries
        """
        if not self.pipeline_agent:
            raise ValueError("Pipeline not initialized. Call initialize_pipeline() first.")
        
        results = []
        
        # Process queries in batches
        for i in range(0, len(queries), max_concurrent):
            batch = queries[i:i+max_concurrent]
            
            # Create tasks for concurrent processing
            tasks = []
            for query_info in batch:
                task = self.pipeline_agent.process_comprehensive_query(
                    user_query=query_info['query'],
                    local_model_id=query_info['model_id']
                )
                tasks.append(task)
            
            # Process batch
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle results and exceptions
            for query_info, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    error_result = {
                        "user_query": query_info['query'],
                        "error": str(result),
                        "processing_time": 0.0
                    }
                    results.append(error_result)
                else:
                    results.append(result)
        
        # Store in history
        self.results_history.extend(results)
        
        return results
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a summary report of all processed queries."""
        if not self.results_history:
            return {"message": "No queries processed yet"}
        
        # Calculate statistics
        total_queries = len(self.results_history)
        successful_queries = len([r for r in self.results_history if not r.get('error')])
        failed_queries = total_queries - successful_queries
        
        avg_processing_time = sum(r.get('processing_time', 0) for r in self.results_history) / total_queries
        
        # Query enhancement statistics
        enhanced_queries = len([
            r for r in self.results_history 
            if r.get('pipeline_steps', {}).get('local_data_retrieval', {}).get('query_was_enriched', False)
        ])
        
        return {
            "summary": {
                "total_queries": total_queries,
                "successful_queries": successful_queries,
                "failed_queries": failed_queries,
                "success_rate": successful_queries / total_queries if total_queries > 0 else 0,
                "average_processing_time": avg_processing_time,
                "queries_enhanced": enhanced_queries,
                "enhancement_rate": enhanced_queries / total_queries if total_queries > 0 else 0
            },
            "performance_metrics": {
                "fastest_query": min(r.get('processing_time', float('inf')) for r in self.results_history),
                "slowest_query": max(r.get('processing_time', 0) for r in self.results_history),
                "median_processing_time": sorted([r.get('processing_time', 0) for r in self.results_history])[total_queries // 2] if total_queries > 0 else 0
            }
        }

if __name__ == "__main__":
    # Run the example
    import asyncio
    asyncio.run(main_example())