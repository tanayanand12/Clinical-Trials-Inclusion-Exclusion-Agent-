# tpaa_clinical_trials_test.py
"""
Specialized testing script for Transpedal Arterial Access (TPAA) clinical trials queries.
This script tests specific technical, procedural, and safety questions related to TPAA procedures.
"""

import os
import logging
from dotenv import load_dotenv # type: ignore
from openai import OpenAI # type: ignore
from clinical_trials_rag_pipeline import ClinicalTrialsRAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Main function testing TPAA-specific clinical trials queries.
    """
    # Load environment variables
    load_dotenv()
    
    # Verify OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found in environment variables")
        return
    
    try:
        # Initialize OpenAI client
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize the Clinical Trials RAG Pipeline with optimized settings for TPAA queries
        logger.info("Initializing Clinical Trials RAG Pipeline for TPAA studies...")
        
        pipeline = ClinicalTrialsRAGPipeline(
            openai_client=openai_client,
            model_name="gpt-4-turbo",  # Better for complex medical queries
            embedding_model="text-embedding-3-small",  # Updated embedding model
            max_trials=25,  # Increased for comprehensive TPAA research
            max_context_length=8000,  # Larger context for detailed medical info
            chunk_size=1000,  # Larger chunks for procedural details
            chunk_overlap=200  # More overlap for medical continuity
        )
        
        # TPAA-specific test queries organized by category
        tpaa_test_queries = [
            # Technical/Procedural Questions
            {
                "category": "Technical/Procedural",
                "queries": [
                    "What is the success rate of transpedal access compared to retrograde tibial access in below-the-knee interventions?",
                    "How many studies have compared 4Fr vs 6Fr sheath sizes for transpedal arterial access procedures?",
                    "What are the reported rates of arterial spasm during transpedal catheterization across published trials?",
                    "Which closure devices (Angio-Seal, Mynx, manual compression) show the lowest complication rates in transpedal access studies?"
                ]
            },
            
            # Patient Population & Selection
            {
                "category": "Patient Population & Selection",
                "queries": [
                    "What percentage of TPAA studies include patients with chronic kidney disease stages 4-5?",
                    "How many trials have specifically enrolled patients with severely calcified tibial vessels for transpedal intervention?",
                    "What is the average ankle-brachial index (ABI) range in patients enrolled in transpedal access studies?",
                    "How many studies have compared outcomes in diabetic vs non-diabetic patients undergoing transpedal procedures?"
                ]
            },
            
            # Complications & Safety
            {
                "category": "Complications & Safety",
                "queries": [
                    "What is the pooled incidence of pseudoaneurysm formation following transpedal arterial access across randomized trials?",
                    "How many studies report on compartment syndrome rates following transpedal access procedures?",
                    "What are the reported rates of pedal artery occlusion at 30-day follow-up in TPAA studies?",
                    "Which studies have evaluated the risk of foot ischemia following transpedal access in patients with single-vessel runoff?"
                ]
            },
            
            # Comparative Effectiveness
            {
                "category": "Comparative Effectiveness",
                "queries": [
                    "How many head-to-head trials compare transpedal vs transradial access for peripheral interventions?",
                    "What are the reported procedure times for transpedal access vs traditional retrograde femoral access in infrapopliteal interventions?",
                    "Which studies have compared radiation exposure between transpedal and femoral access approaches?",
                    "How many trials have evaluated cost-effectiveness of transpedal vs alternative access routes?"
                ]
            },
            
            # Device & Technology Specific
            {
                "category": "Device & Technology Specific",
                "queries": [
                    "What balloon catheter sizes (2.0mm, 2.5mm, 3.0mm) are most commonly used in transpedal angioplasty studies?",
                    "How many studies have evaluated drug-coated balloons via transpedal access for infrapopliteal disease?",
                    "Which trials have assessed the use of atherectomy devices through transpedal arterial access?",
                    "What are the reported crossing rates for chronic total occlusions using transpedal access in published studies?"
                ]
            }
        ]
        
        # Process each category and query
        total_queries = sum(len(category["queries"]) for category in tpaa_test_queries)
        current_query = 0
        
        print(f"\n{'='*100}")
        print(f"TRANSPEDAL ARTERIAL ACCESS (TPAA) CLINICAL TRIALS ANALYSIS")
        print(f"Testing {total_queries} specialized queries across {len(tpaa_test_queries)} categories")
        print('='*100)
        
        results_summary = {
            "successful_queries": 0,
            "failed_queries": 0,
            "categories_tested": len(tpaa_test_queries),
            "total_studies_found": set(),
            "processing_times": []
        }
        
        for category_data in tpaa_test_queries:
            category = category_data["category"]
            queries = category_data["queries"]
            
            print(f"\n{'#'*80}")
            print(f"CATEGORY: {category.upper()}")
            print(f"Testing {len(queries)} queries in this category")
            print('#'*80)
            
            category_results = {
                "successful": 0,
                "failed": 0,
                "studies_found": set()
            }
            
            for i, query in enumerate(queries, 1):
                current_query += 1
                
                print(f"\n{'-'*60}")
                print(f"Query {current_query}/{total_queries} (Category {category} - Q{i})")
                print(f"Question: {query}")
                print('-'*60)
                
                try:
                    # Process the TPAA-specific query
                    result = pipeline.process_query(
                        query=query,
                        top_k=10  # More relevant chunks for specialized queries
                    )
                    
                    if result['success']:
                        print("✅ SUCCESS!")
                        print(f"\nAnswer:\n{result['answer']}")
                        
                        # Track studies found
                        study_ids = {study['study_id'] for study in result['studies']}
                        category_results["studies_found"].update(study_ids)
                        results_summary["total_studies_found"].update(study_ids)
                        
                        # Display top relevant studies
                        if result['studies']:
                            print(f"\nRelevant Studies Found: {len(result['studies'])}")
                            print("Top 3 Most Relevant:")
                            for j, study in enumerate(result['studies'][:3], 1):
                                print(f"  {j}. NCT{study['study_id']}: {study['title'][:80]}...")
                                print(f"     Relevance Score: {study['similarity_score']:.3f}")
                        
                        # Processing metrics
                        metadata = result['metadata']
                        processing_time = metadata['processing_time']
                        results_summary["processing_times"].append(processing_time)
                        
                        print(f"\nMetrics:")
                        print(f"  • Processing time: {processing_time:.2f}s")
                        print(f"  • Trials analyzed: {metadata['trials_processed']}")
                        print(f"  • Chunks analyzed: {metadata['chunks_created']}")
                        print(f"  • Quality score: {result['quality_assessment']['overall_score']:.2f}")
                        
                        category_results["successful"] += 1
                        results_summary["successful_queries"] += 1
                        
                    else:
                        print(f"❌ FAILED: {result['error']}")
                        category_results["failed"] += 1
                        results_summary["failed_queries"] += 1
                        
                except Exception as e:
                    print(f"⚠️ ERROR: {str(e)}")
                    logger.error(f"Error processing TPAA query '{query}': {e}")
                    category_results["failed"] += 1
                    results_summary["failed_queries"] += 1
            
            # Category summary
            print(f"\n{'='*60}")
            print(f"CATEGORY '{category}' SUMMARY:")
            print(f"  • Successful queries: {category_results['successful']}/{len(queries)}")
            print(f"  • Failed queries: {category_results['failed']}/{len(queries)}")
            print(f"  • Unique studies found: {len(category_results['studies_found'])}")
            if category_results['studies_found']:
                print(f"  • Study IDs: {', '.join(sorted(category_results['studies_found']))}")
            print('='*60)
        
        # Final comprehensive summary
        print(f"\n{'#'*100}")
        print("FINAL TPAA CLINICAL TRIALS ANALYSIS SUMMARY")
        print('#'*100)
        
        success_rate = (results_summary["successful_queries"] / total_queries) * 100
        avg_processing_time = sum(results_summary["processing_times"]) / len(results_summary["processing_times"]) if results_summary["processing_times"] else 0
        
        print(f"📊 OVERALL RESULTS:")
        print(f"  • Total queries tested: {total_queries}")
        print(f"  • Successful queries: {results_summary['successful_queries']} ({success_rate:.1f}%)")
        print(f"  • Failed queries: {results_summary['failed_queries']}")
        print(f"  • Categories covered: {results_summary['categories_tested']}")
        print(f"  • Unique studies identified: {len(results_summary['total_studies_found'])}")
        print(f"  • Average processing time: {avg_processing_time:.2f} seconds")
        
        print(f"\n🔬 STUDY COVERAGE:")
        if results_summary['total_studies_found']:
            print(f"  • TPAA-related studies found: {', '.join(sorted(results_summary['total_studies_found']))}")
        else:
            print("  • No specific TPAA studies found - may need broader search terms")
        
        print(f"\n📈 PERFORMANCE ANALYSIS:")
        if success_rate >= 80:
            print("  • Excellent performance: Pipeline handles TPAA queries very well")
        elif success_rate >= 60:
            print("  • Good performance: Most TPAA queries processed successfully")
        elif success_rate >= 40:
            print("  • Moderate performance: Some TPAA queries may need refinement")
        else:
            print("  • Performance needs improvement: Consider expanding data sources")
        
        # Pipeline status
        final_status = pipeline.get_pipeline_status()
        print(f"\n🔧 PIPELINE STATUS:")
        for component, status in final_status['components'].items():
            print(f"  • {component}: {'✅' if status else '❌'}")
        
        print(f"\n{'#'*100}")
        print("TPAA CLINICAL TRIALS TESTING COMPLETE")
        print('#'*100)
        
    except Exception as e:
        logger.error(f"Failed to run TPAA testing: {e}")
        print(f"❌ TPAA testing failed: {e}")

def interactive_tpaa_mode():
    """
    Interactive mode specifically for TPAA queries.
    """
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables")
        return
    
    try:
        # Initialize pipeline
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        pipeline = ClinicalTrialsRAGPipeline(
            openai_client=openai_client,
            max_trials=25,
            max_context_length=8000
        )
        
        print("🦶 TPAA Clinical Trials RAG Pipeline - Interactive Mode")
        print("Specialized for Transpedal Arterial Access queries")
        print("Type 'quit' to exit, 'help' for TPAA-specific guidance")
        print("-" * 70)
        
        while True:
            query = input("\n🔍 Enter your TPAA clinical question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif query.lower() == 'help':
                print("\n📋 TPAA-specific questions you can ask:")
                print("• Success rates of transpedal vs other access methods")
                print("• Complication rates (pseudoaneurysm, spasm, occlusion)")
                print("• Device comparisons (sheath sizes, closure devices)")
                print("• Patient selection criteria and outcomes")
                print("• Procedural details (balloon sizes, crossing rates)")
                print("• Safety profiles in specific populations")
                continue
            elif not query:
                print("Please enter a question.")
                continue
            
            print(f"\n🔄 Processing TPAA query: {query}")
            print("Please wait...")
            
            try:
                result = pipeline.process_query(query, top_k=10)
                
                if result['success']:
                    print(f"\n✅ Answer:\n{result['answer']}")
                    print(f"\n📊 Based on {len(result['studies'])} relevant studies")
                    print(f"⏱️ Processing time: {result['metadata']['processing_time']:.2f} seconds")
                    
                    if result['studies']:
                        print(f"\n🔬 Top relevant studies:")
                        for i, study in enumerate(result['studies'][:3], 1):
                            print(f"  {i}. NCT{study['study_id']}: {study['title'][:60]}...")
                else:
                    print(f"\n❌ Error: {result['error']}")
                    
            except Exception as e:
                print(f"\n⚠️ Error processing query: {e}")
    
    except Exception as e:
        print(f"Failed to initialize TPAA interactive mode: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_tpaa_mode()
    else:
        main()