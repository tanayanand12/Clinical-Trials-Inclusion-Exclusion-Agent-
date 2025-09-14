# transpedal_access_example.py
"""
Example usage of the Clinical Trials RAG Pipeline for Transpedal Arterial Access research.
This script demonstrates how to use the pipeline to query clinical trials data specifically
focused on transpedal access procedures and related interventions.

All processes, responses, and analysis results are logged to specified files for later analysis.
"""

import os
import logging
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv # type: ignore
from openai import OpenAI # type: ignore
from clinical_trials_rag_pipeline import ClinicalTrialsRAGPipeline

def setup_logging(log_dir="tpaa_analysis_logs", log_filename=None):
    """
    Set up comprehensive logging for TPAA analysis.
    
    Args:
        log_dir (str): Directory to store log files
        log_filename (str): Specific log filename (optional)
        
    Returns:
        tuple: (logger, log_files_info)
    """
    # Create log directory
    Path(log_dir).mkdir(exist_ok=True)
    
    # Generate timestamp for unique file names if no specific filename provided
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define log file paths
    if log_filename:
        base_name = log_filename.replace('.log', '')
        log_files = {
            'main_log': f"{log_dir}/{base_name}.log",
            'results_json': f"{log_dir}/{base_name}_results.json",
            'detailed_log': f"{log_dir}/{base_name}_detailed.log",
            'summary_log': f"{log_dir}/{base_name}_summary.txt"
        }
    else:
        log_files = {
            'main_log': f"{log_dir}/tpaa_analysis_{timestamp}.log",
            'results_json': f"{log_dir}/tpaa_results_{timestamp}.json",
            'detailed_log': f"{log_dir}/tpaa_detailed_{timestamp}.log",
            'summary_log': f"{log_dir}/tpaa_summary_{timestamp}.txt"
        }
    
    # Configure main logger
    logger = logging.getLogger('TPAA_Analysis')
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler for main log
    main_handler = logging.FileHandler(log_files['main_log'])
    main_handler.setLevel(logging.INFO)
    main_handler.setFormatter(simple_formatter)
    logger.addHandler(main_handler)
    
    # File handler for detailed log
    detailed_handler = logging.FileHandler(log_files['detailed_log'])
    detailed_handler.setLevel(logging.DEBUG)
    detailed_handler.setFormatter(detailed_formatter)
    logger.addHandler(detailed_handler)
    
    # Console handler (optional - can be disabled)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # Log the initialization
    logger.info("="*80)
    logger.info("TPAA Clinical Trials Analysis Session Started")
    logger.info(f"Session ID: {timestamp}")
    logger.info(f"Log files created:")
    for log_type, path in log_files.items():
        logger.info(f"  {log_type}: {path}")
    logger.info("="*80)
    
    return logger, log_files

def log_query_result(logger, log_files, query_num, query_data, result, error=None):
    """
    Log detailed query results to multiple formats.
    
    Args:
        logger: Main logger instance
        log_files (dict): Dictionary of log file paths
        query_num (int): Query number
        query_data (dict): Query information
        result (dict): Query result from pipeline
        error (str): Error message if any
    """
    timestamp = datetime.now().isoformat()
    
    # Log to main logger
    logger.info(f"QUERY {query_num}: {query_data['category']}")
    logger.info(f"Question: {query_data['query']}")
    
    if error:
        logger.error(f"ERROR: {error}")
        return
    
    if result['success']:
        studies_count = len(result['studies'])
        processing_time = result['metadata']['processing_time']
        quality_score = result['quality_assessment']['overall_score']
        
        logger.info(f"SUCCESS: {studies_count} studies analyzed in {processing_time:.2f}s")
        logger.info(f"Quality Score: {quality_score:.3f}")
        logger.info(f"Answer Preview: {result['answer'][:100]}...")
    else:
        logger.error(f"FAILED: {result['error']}")
    
    # Prepare detailed result for JSON logging
    detailed_result = {
        'timestamp': timestamp,
        'query_number': query_num,
        'category': query_data['category'],
        'description': query_data['description'],
        'query': query_data['query'],
        'success': result['success'] if not error else False,
        'error': error or result.get('error'),
        'processing_time': result['metadata']['processing_time'] if result['success'] else None,
        'studies_analyzed': len(result['studies']) if result['success'] else 0,
        'quality_score': result['quality_assessment']['overall_score'] if result['success'] else None,
        'answer': result['answer'] if result['success'] else None,
        'metadata': result['metadata'] if result['success'] else None,
        'studies': [
            {
                'study_id': study['study_id'],
                'title': study['title'],
                'similarity_score': study['similarity_score']
            } for study in result['studies'][:5]  # Top 5 studies
        ] if result['success'] else []
    }
    
    # Append to JSON results file
    try:
        # Read existing results
        json_results = []
        if os.path.exists(log_files['results_json']):
            with open(log_files['results_json'], 'r', encoding='utf-8') as f:
                json_results = json.load(f)
        
        # Add new result
        json_results.append(detailed_result)
        
        # Write back to file
        with open(log_files['results_json'], 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        logger.error(f"Failed to write JSON results: {e}")

def create_summary_report(logger, log_files, total_queries, successful_queries, failed_queries, start_time, pipeline_status):
    """
    Create a comprehensive summary report.
    """
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    
    summary_content = f"""
TRANSPEDAL ARTERIAL ACCESS (TPAA) CLINICAL TRIALS ANALYSIS SUMMARY
================================================================

Analysis Session Information:
- Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}
- End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
- Total Duration: {total_duration:.2f} seconds ({total_duration/60:.1f} minutes)

Query Processing Results:
- Total Queries Processed: {total_queries}
- Successful Analyses: {successful_queries}
- Failed Analyses: {failed_queries}
- Success Rate: {(successful_queries/total_queries*100):.1f}%

Pipeline Status:
"""
    
    for component, status in pipeline_status['components'].items():
        status_text = "✓ ACTIVE" if status else "✗ INACTIVE"
        summary_content += f"- {component}: {status_text}\n"
    
    summary_content += f"""
Research Categories Analyzed:
- Comparative Effectiveness
- Equipment Analysis  
- Complications Assessment
- Patient Demographics
- Procedural Efficiency
- Device Utilization
- Follow-up Outcomes

Log Files Generated:
- Main Log: {log_files['main_log']}
- Detailed Log: {log_files['detailed_log']}
- Results JSON: {log_files['results_json']}
- Summary Report: {log_files['summary_log']}

Analysis Complete: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
================================================================
"""
    
    # Write summary to file
    with open(log_files['summary_log'], 'w', encoding='utf-8') as f:
        f.write(summary_content)
    
    logger.info("Summary report created successfully")
    print(f"\n📊 Analysis complete! Summary saved to: {log_files['summary_log']}")

def main(log_filename=None):
    """
    Main function demonstrating the Clinical Trials RAG Pipeline usage for transpedal access research.
    
    Args:
        log_filename (str): Optional specific log filename
    """
    start_time = datetime.now()
    
    # Setup comprehensive logging
    logger, log_files = setup_logging(log_filename=log_filename)
    
    # Load environment variables
    load_dotenv()
    
    # Verify OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found in environment variables")
        return
    
    successful_queries = 0
    failed_queries = 0
    
    try:
        # Initialize OpenAI client
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize the Clinical Trials RAG Pipeline with optimized settings for vascular research
        logger.info("Initializing Clinical Trials RAG Pipeline for Transpedal Access Research...")
        
        pipeline = ClinicalTrialsRAGPipeline(
            openai_client=openai_client,
            model_name="gpt-4-turbo",  # Use GPT-4 for better medical analysis
            embedding_model="text-embedding-ada-002",
            max_trials=25,  # Increased for comprehensive vascular research
            max_context_length=8000,  # Larger context for detailed medical data
            chunk_size=1000,  # Larger chunks for medical procedures
            chunk_overlap=200   # More overlap for continuity
        )
        
        # Check pipeline status
        status = pipeline.get_pipeline_status()
        logger.info(f"Pipeline initialization status: {status['pipeline_initialized']}")
        
        # Transpedal Arterial Access (TPAA) research queries
        tpaa_queries = [
            # Success rates and comparative effectiveness
            {
                "query": "What is the success rate of transpedal access compared to retrograde tibial access in below-the-knee interventions?",
                "category": "Comparative Effectiveness",
                "description": "Success rate comparison between access methods"
            },
            
            # Equipment and technique comparisons
            {
                "query": "How many studies have compared 4Fr vs 6Fr sheath sizes for transpedal arterial access procedures?",
                "category": "Equipment Analysis",
                "description": "Sheath size comparison studies"
            },
            
            # Complications and adverse events
            {
                "query": "What are the reported rates of arterial spasm during transpedal catheterization across published trials?",
                "category": "Complications",
                "description": "Arterial spasm incidence rates"
            },
            
            # Closure devices and techniques
            {
                "query": "Which closure devices (Angio-Seal, Mynx, manual compression) show the lowest complication rates in transpedal access studies?",
                "category": "Closure Techniques",
                "description": "Closure device safety comparison"
            },
            
            # Patient population characteristics
            {
                "query": "What percentage of TPAA studies include patients with chronic kidney disease stages 4-5?",
                "category": "Patient Demographics",
                "description": "CKD patient inclusion rates"
            },
            
            # Vessel characteristics and patient selection
            {
                "query": "How many trials have specifically enrolled patients with severely calcified tibial vessels for transpedal intervention?",
                "category": "Patient Selection",
                "description": "Calcified vessel patient enrollment"
            },
            
            # Hemodynamic measurements
            {
                "query": "What is the average ankle-brachial index (ABI) range in patients enrolled in transpedal access studies?",
                "category": "Hemodynamics",
                "description": "ABI ranges in study populations"
            },
            
            # Diabetic patient outcomes
            {
                "query": "How many studies have compared outcomes in diabetic vs non-diabetic patients undergoing transpedal procedures?",
                "category": "Diabetes Outcomes",
                "description": "Diabetic vs non-diabetic comparisons"
            },
            
            # Specific complications
            {
                "query": "What is the pooled incidence of pseudoaneurysm formation following transpedal arterial access across randomized trials?",
                "category": "Vascular Complications",
                "description": "Pseudoaneurysm incidence rates"
            },
            
            # Severe complications
            {
                "query": "How many studies report on compartment syndrome rates following transpedal access procedures?",
                "category": "Severe Complications",
                "description": "Compartment syndrome reporting"
            },
            
            # Follow-up outcomes
            {
                "query": "What are the reported rates of pedal artery occlusion at 30-day follow-up in TPAA studies?",
                "category": "Follow-up Outcomes",
                "description": "30-day occlusion rates"
            },
            
            # Risk stratification
            {
                "query": "Which studies have evaluated the risk of foot ischemia following transpedal access in patients with single-vessel runoff?",
                "category": "Risk Assessment",
                "description": "Ischemia risk in single-vessel runoff"
            },
            
            # Access route comparisons
            {
                "query": "How many head-to-head trials compare transpedal vs transradial access for peripheral interventions?",
                "category": "Access Comparison",
                "description": "Transpedal vs transradial trials"
            },
            
            # Procedural efficiency
            {
                "query": "What are the reported procedure times for transpedal access vs traditional retrograde femoral access in infrapopliteal interventions?",
                "category": "Procedural Efficiency",
                "description": "Procedure time comparisons"
            },
            
            # Radiation exposure
            {
                "query": "Which studies have compared radiation exposure between transpedal and femoral access approaches?",
                "category": "Radiation Safety",
                "description": "Radiation exposure comparisons"
            },
            
            # Health economics
            {
                "query": "How many trials have evaluated cost-effectiveness of transpedal vs alternative access routes?",
                "category": "Health Economics",
                "description": "Cost-effectiveness analyses"
            },
            
            # Intervention devices - balloons
            {
                "query": "What balloon catheter sizes (2.0mm, 2.5mm, 3.0mm) are most commonly used in transpedal angioplasty studies?",
                "category": "Intervention Devices",
                "description": "Balloon catheter size utilization"
            },
            
            # Drug-coated devices
            {
                "query": "How many studies have evaluated drug-coated balloons via transpedal access for infrapopliteal disease?",
                "category": "Drug-Coated Devices",
                "description": "DCB usage in transpedal access"
            },
            
            # Atherectomy devices
            {
                "query": "Which trials have assessed the use of atherectomy devices through transpedal arterial access?",
                "category": "Atherectomy",
                "description": "Atherectomy device utilization"
            },
            
            # Complex lesions
            {
                "query": "What are the reported crossing rates for chronic total occlusions using transpedal access in published studies?",
                "category": "Complex Lesions",
                "description": "CTO crossing success rates"
            }
        ]
        
        # Process each transpedal access query
        logger.info("Starting analysis of 20 TPAA research queries")
        print(f"\n{'='*100}")
        print("TRANSPEDAL ARTERIAL ACCESS (TPAA) CLINICAL TRIALS ANALYSIS")
        print('='*100)
        
        for i, example in enumerate(tpaa_queries, 1):
            logger.info(f"Processing query {i}/20: {example['category']}")
            print(f"\n{'='*100}")
            print(f"QUERY {i}/20: {example['category']}")
            print('='*100)
            print(f"Question: {example['query']}")
            print(f"Focus: {example['description']}")
            print('-'*100)
            
            try:
                # Process the query with increased top_k for comprehensive medical research
                result = pipeline.process_query(
                    query=example['query'],
                    top_k=12  # Use more relevant chunks for thorough medical analysis
                )
                
                # Log the result
                log_query_result(logger, log_files, i, example, result)
                
                if result['success']:
                    successful_queries += 1
                    # Display results
                    print("✅ ANALYSIS COMPLETE")
                    print(f"\nFindings:\n{result['answer']}")
                    
                    # Study statistics
                    studies_count = len(result['studies'])
                    print(f"\n📊 Research Base: {studies_count} relevant studies analyzed")
                    
                    if result['studies']:
                        print(f"\n🔬 Key Studies Identified:")
                        for j, study in enumerate(result['studies'][:5], 1):  # Show top 5
                            title = study['title'][:80] + "..." if len(study['title']) > 80 else study['title']
                            print(f"  {j}. {title}")
                            print(f"     📋 NCT ID: {study['study_id']} | Relevance: {study['similarity_score']:.3f}")
                    
                    # Processing metrics
                    metadata = result['metadata']
                    print(f"\n⚡ Processing Metrics:")
                    print(f"  • Analysis time: {metadata['processing_time']:.2f} seconds")
                    print(f"  • Trials screened: {metadata['total_trials_fetched']}")
                    print(f"  • Trials analyzed: {metadata['trials_processed']}")
                    print(f"  • Data chunks processed: {metadata['chunks_created']}")
                    print(f"  • Relevant evidence segments: {metadata['relevant_chunks']}")
                    
                    # Quality assessment
                    quality = result['quality_assessment']
                    confidence_emoji = "🟢" if quality['overall_score'] >= 0.8 else "🟡" if quality['overall_score'] >= 0.6 else "🔴"
                    print(f"  • Evidence quality: {quality['overall_score']:.2f} {confidence_emoji} ({quality['quality_level']})")
                    
                else:
                    failed_queries += 1
                    print(f"❌ ANALYSIS FAILED: {result['error']}")
                    
            except Exception as e:
                failed_queries += 1
                error_msg = str(e)
                print(f"💥 ERROR: {error_msg}")
                logger.error(f"Error processing TPAA query '{example['query']}': {error_msg}")
                log_query_result(logger, log_files, i, example, None, error=error_msg)
        
        # Get final pipeline status
        final_status = pipeline.get_pipeline_status()
        
        # Create comprehensive summary report
        create_summary_report(logger, log_files, len(tpaa_queries), successful_queries, failed_queries, start_time, final_status)
        
        print(f"\n{'='*100}")
        print("🎯 TRANSPEDAL ACCESS RESEARCH ANALYSIS COMPLETE")
        print('='*100)
        
        # Show final pipeline statistics
        print(f"\n📈 Final Pipeline Status:")
        for component, status in final_status['components'].items():
            status_icon = '✅' if status else '❌'
            print(f"  • {component}: {status_icon}")
        
        print(f"\n💡 Research Summary:")
        print(f"  • Total queries processed: {len(tpaa_queries)}")
        print(f"  • Successful analyses: {successful_queries}")
        print(f"  • Failed analyses: {failed_queries}")
        print(f"  • Success rate: {(successful_queries/len(tpaa_queries)*100):.1f}%")
        print(f"  • Focus areas covered: {len(set(q['category'] for q in tpaa_queries))}")
        
        logger.info("Analysis session completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to run TPAA research demonstration: {e}")
        print(f"❌ Demonstration failed: {e}")

def interactive_tpaa_mode(log_filename=None):
    """
    Interactive mode for custom transpedal access queries with logging.
    """
    # Setup logging
    logger, log_files = setup_logging(log_filename=log_filename)
    
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found in environment variables")
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        return
    
    try:
        # Initialize pipeline
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        pipeline = ClinicalTrialsRAGPipeline(
            openai_client=openai_client,
            max_trials=20,
            max_context_length=7000
        )
        
        logger.info("Interactive TPAA mode started")
        print("🩺 Transpedal Arterial Access Clinical Trials - Interactive Research Mode")
        print("Type 'quit' to exit, 'examples' for sample questions")
        print("-" * 70)
        
        query_count = 0
        
        while True:
            query = input("\n🔍 Enter your transpedal access research question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                logger.info(f"Interactive session ended. Total queries processed: {query_count}")
                print("👋 Thank you for using the TPAA research tool!")
                break
            elif query.lower() in ['examples', 'help']:
                print("\n💡 Example transpedal access research questions:")
                print("• What are the success rates of transpedal vs femoral access?")
                print("• How many studies compare different sheath sizes for transpedal access?")
                print("• What are the complication rates of transpedal procedures?")
                print("• Which closure techniques work best for transpedal access?")
                print("• What patient populations are studied in transpedal trials?")
                print("• How do outcomes differ in diabetic patients?")
                print("• What devices are used through transpedal access?")
                continue
            elif not query:
                print("Please enter a research question.")
                continue
            
            query_count += 1
            logger.info(f"Interactive query {query_count}: {query}")
            
            print(f"\n🔬 Analyzing: {query}")
            print("🔄 Searching clinical trials database...")
            
            try:
                result = pipeline.process_query(query, top_k=10)
                
                # Log the interactive result
                query_data = {
                    'query': query,
                    'category': 'Interactive Query',
                    'description': 'User-submitted query in interactive mode'
                }
                log_query_result(logger, log_files, query_count, query_data, result)
                
                if result['success']:
                    print(f"\n📋 Research Findings:\n{result['answer']}")
                    
                    studies_count = len(result['studies'])
                    processing_time = result['metadata']['processing_time']
                    quality_score = result['quality_assessment']['overall_score']
                    
                    print(f"\n📊 Analysis Summary:")
                    print(f"  • Studies analyzed: {studies_count}")
                    print(f"  • Processing time: {processing_time:.2f} seconds")
                    print(f"  • Evidence quality: {quality_score:.2f}")
                    
                    if studies_count > 0:
                        print(f"\n🔬 Top relevant studies:")
                        for i, study in enumerate(result['studies'][:3], 1):
                            title = study['title'][:60] + "..." if len(study['title']) > 60 else study['title']
                            print(f"  {i}. {title} (NCT: {study['study_id']})")
                else:
                    print(f"\n❌ Error: {result['error']}")
                    
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error in interactive query {query_count}: {error_msg}")
                print(f"\n💥 Error processing query: {error_msg}")
    
    except Exception as e:
        logger.error(f"Failed to initialize interactive mode: {e}")
        print(f"❌ Failed to initialize interactive mode: {e}")

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    log_filename = None
    mode = "main"
    
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "interactive":
            mode = "interactive"
        elif arg.startswith("--log="):
            log_filename = arg.split("=", 1)[1]
        elif arg == "--log" and i < len(sys.argv) - 1:
            log_filename = sys.argv[i + 1]
    
    if mode == "interactive":
        interactive_tpaa_mode(log_filename=log_filename)
    else:
        main(log_filename=log_filename)
    
    print(f"\n📝 All analysis results have been saved to log files in the 'tpaa_analysis_logs' directory")
    print("   You can review the detailed results later using the JSON and text log files.")