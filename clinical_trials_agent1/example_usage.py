# example_usage.py
"""
Example usage of the Clinical Trials RAG Pipeline.
This script demonstrates how to use the pipeline to query clinical trials data.
"""

import os
import logging
from dotenv import load_dotenv # type: ignore
from openai import OpenAI # type: ignore
from .clinical_trials_rag_pipeline import ClinicalTrialsRAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Main function demonstrating the Clinical Trials RAG Pipeline usage.
    """
    # Load environment variables
    load_dotenv()
    
    # Verify OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found in environment variables")
        return
    
    try:
        # Initialize OpenAI client (optional - pipeline can work without it for manual URL construction)
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize the Clinical Trials RAG Pipeline
        logger.info("Initializing Clinical Trials RAG Pipeline...")
        
        pipeline = ClinicalTrialsRAGPipeline(
            openai_client=openai_client,
            model_name="gpt-4-turbo",  # or "gpt-3.5-turbo" for faster/cheaper responses
            embedding_model="text-embedding-ada-002",
            max_trials=15,  # Limit for demonstration
            max_context_length=6000,  # Reduced for faster processing
            chunk_size=800,  # Smaller chunks for better granularity
            chunk_overlap=150
        )
        
        # Check pipeline status
        status = pipeline.get_pipeline_status()
        logger.info(f"Pipeline status: {status['pipeline_initialized']}")
        
        # Example queries to demonstrate different types of questions
        example_queries = [
            # Count/statistical query
            {
                "query": "How many recruiting clinical trials are there for Type 2 Diabetes in India?",
                "description": "Statistical query about trial counts"
            },
            
            # Intervention/treatment query
            {
                "query": "What interventions are being tested for diabetes treatment in current trials?",
                "description": "Query about specific interventions"
            },
            
            # Eligibility criteria query
            {
                "query": "What are the common eligibility criteria for diabetes clinical trials?",
                "description": "Query about patient selection criteria"
            },
            
            # Location-based query
            {
                "query": "Which hospitals in India are conducting diabetes research studies?",
                "description": "Location-specific query"
            },
            
            # Outcome measures query
            {
                "query": "What are the primary endpoints measured in diabetes trials?",
                "description": "Query about study outcomes"
            }
        ]
        
        # Process each example query
        for i, example in enumerate(example_queries, 1):
            print(f"\n{'='*80}")
            print(f"EXAMPLE {i}: {example['description']}")
            print('='*80)
            print(f"Query: {example['query']}")
            print('-'*80)
            
            try:
                # Process the query
                result = pipeline.process_query(
                    query=example['query'],
                    top_k=8  # Use top 8 most relevant chunks
                )
                
                if result['success']:
                    # Display results
                    print("SUCCESS!")
                    print(f"\nAnswer:\n{result['answer']}")
                    
                    print(f"\nStudies Analyzed: {len(result['studies'])}")
                    if result['studies']:
                        print("\nTop Relevant Studies:")
                        for j, study in enumerate(result['studies'][:3], 1):
                            print(f"  {j}. {study['title'][:100]}...")
                            print(f"     NCT ID: {study['study_id']}, Relevance: {study['similarity_score']:.3f}")
                    
                    # Display metadata
                    metadata = result['metadata']
                    print(f"\nProcessing Metrics:")
                    print(f"  • Total processing time: {metadata['processing_time']:.2f} seconds")
                    print(f"  • Trials fetched: {metadata['total_trials_fetched']}")
                    print(f"  • Trials processed: {metadata['trials_processed']}")
                    print(f"  • Chunks created: {metadata['chunks_created']}")
                    print(f"  • Relevant chunks used: {metadata['relevant_chunks']}")
                    
                    # Display quality assessment
                    quality = result['quality_assessment']
                    print(f"  • Response quality: {quality['overall_score']:.2f} ({quality['quality_level']})")
                    
                else:
                    print(f"FAILED: {result['error']}")
                    
            except Exception as e:
                print(f"ERROR processing query: {e}")
                logger.error(f"Error processing query '{example['query']}': {e}")
        
        print(f"\n{'='*80}")
        print("DEMONSTRATION COMPLETE")
        print('='*80)
        
        # Show final pipeline statistics
        print(f"\nFinal Pipeline Status:")
        final_status = pipeline.get_pipeline_status()
        for component, status in final_status['components'].items():
            print(f"  • {component}: {'✓' if status else '✗'}")
        
    except Exception as e:
        logger.error(f"Failed to run demonstration: {e}")
        print(f"Demonstration failed: {e}")

def interactive_mode():
    """
    Interactive mode for testing custom queries.
    """
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables")
        return
    
    try:
        # Initialize pipeline
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        pipeline = ClinicalTrialsRAGPipeline(openai_client=openai_client)
        
        print("Clinical Trials RAG Pipeline - Interactive Mode")
        print("Type 'quit' to exit, 'help' for guidance")
        print("-" * 50)
        
        while True:
            query = input("\nEnter your clinical trials question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            elif query.lower() == 'help':
                print("\nExample questions you can ask:")
                print("• How many trials are recruiting for [condition] in [location]?")
                print("• What interventions are being tested for [disease]?")
                print("• What are the eligibility criteria for [condition] trials?")
                print("• Which hospitals are conducting [disease] research?")
                print("• What are the primary endpoints in [condition] studies?")
                continue
            elif not query:
                print("Please enter a question.")
                continue
            
            print(f"\nProcessing: {query}")
            print("Please wait...")
            
            try:
                result = pipeline.process_query(query)
                
                if result['success']:
                    print(f"\nAnswer:\n{result['answer']}")
                    print(f"\nBased on analysis of {len(result['studies'])} relevant studies")
                    print(f"Processing time: {result['metadata']['processing_time']:.2f} seconds")
                else:
                    print(f"\nError: {result['error']}")
                    
            except Exception as e:
                print(f"\nError processing query: {e}")
    
    except Exception as e:
        print(f"Failed to initialize interactive mode: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_mode()
    else:
        main()