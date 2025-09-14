# api.py
import os
import sys
from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime
import logging
from contextlib import asynccontextmanager

# Add root path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from dotenv import load_dotenv # type: ignore
load_dotenv()

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from pydantic import BaseModel, Field #type: ignore
import uvicorn # type: ignore

# Import pipeline components
from src.pipeline import ComprehensiveRAGPipelineAgent, RAGPipelineManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global pipeline manager
pipeline_manager = RAGPipelineManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the pipeline on startup."""
    try:
        # Initialize pipeline with default configuration
        pipeline_manager.initialize_pipeline()
        logger.info("RAG Pipeline API initialized successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {str(e)}")
        raise
    finally:
        logger.info("RAG Pipeline API shutting down")

# Initialize FastAPI app
app = FastAPI(
    title="Clinical Trials Exclusion-Inclusion Creiteria Agent API",
    description="Comprehensive RAG Data Pipeline for Clinical Trials Analysis with Inclusion/Exclusion Criteria",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    user_query: str = Field(..., description="The user's query about clinical trials")
    local_model_id: str = Field(..., description="Model ID for local data retrieval")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context parameters")

class BatchQueryRequest(BaseModel):
    queries: List[Dict[str, str]] = Field(..., description="List of queries with 'query' and 'model_id' keys")
    max_concurrent: int = Field(3, description="Maximum concurrent processing")

class PipelineConfigRequest(BaseModel):
    model_name: str = Field("gpt-4-turbo", description="OpenAI model name")
    max_context_length: int = Field(8000, description="Maximum context length")
    local_data_top_k: int = Field(8, description="Number of local data chunks to retrieve")
    criteria_top_k: int = Field(10, description="Number of criteria results to retrieve")

class QueryResponse(BaseModel):
    user_query: str
    local_model_id: str
    pipeline_steps: Dict[str, Any]
    final_response: Dict[str, Any]
    processing_time: float
    error: Optional[str] = None

class ApiResponse(BaseModel):
    success: bool
    data: Any = None
    message: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

# API Endpoints

@app.get("/", response_model=ApiResponse)
async def root():
    """Root endpoint with API information."""
    return ApiResponse(
        success=True,
        data={
            "name": "Clinical Trials RAG Pipeline API",
            "version": "1.0.0",
            "endpoints": [
                "/docs - API documentation",
                "/query - Process single query",
                "/batch-query - Process multiple queries",
                "/pipeline-summary - Get pipeline configuration",
                # "/results-history - Get processing history",
                # "/summary-report - Get summary statistics"
            ]
        },
        message="Clinical Trials RAG Pipeline API is running"
    )

@app.get("/health", response_model=ApiResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check if pipeline is initialized
        if pipeline_manager.pipeline_agent is None:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        # Test OpenAI connection
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=503, detail="OpenAI API key not configured")
        
        return ApiResponse(
            success=True,
            data={"status": "healthy", "pipeline_initialized": True},
            message="All systems operational"
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.post("/initialize-pipeline", response_model=ApiResponse)
async def initialize_pipeline(config: PipelineConfigRequest):
    """Initialize or reconfigure the pipeline."""
    try:
        pipeline_manager.initialize_pipeline(
            model_name=config.model_name,
            max_context_length=config.max_context_length,
            local_data_top_k=config.local_data_top_k,
            criteria_top_k=config.criteria_top_k
        )
        
        return ApiResponse(
            success=True,
            data=config.dict(),
            message="Pipeline initialized successfully"
        )
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize pipeline: {str(e)}")

@app.get("/pipeline-summary", response_model=ApiResponse)
async def get_pipeline_summary():
    """Get pipeline configuration and capabilities."""
    try:
        if pipeline_manager.pipeline_agent is None:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        summary = pipeline_manager.pipeline_agent.get_pipeline_summary()
        
        return ApiResponse(
            success=True,
            data=summary,
            message="Pipeline summary retrieved successfully"
        )
    except Exception as e:
        logger.error(f"Failed to get pipeline summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline summary: {str(e)}")

@app.post("/query", response_model=ApiResponse)
async def process_query(request: QueryRequest):
    """Process a single comprehensive query through the RAG pipeline."""
    try:
        if pipeline_manager.pipeline_agent is None:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        logger.info(f"Processing query: {request.user_query[:100]}...")
        
        result = await pipeline_manager.pipeline_agent.process_comprehensive_query(
            user_query=request.user_query,
            local_model_id=request.local_model_id,
            context=request.context
        )
        
        # Add to history
        pipeline_manager.results_history.append(result)
        
        return ApiResponse(
            success=True,
            data=result,
            message="Query processed successfully"
        )
    except Exception as e:
        logger.error(f"Failed to process query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")

@app.post("/batch-query", response_model=ApiResponse)
async def process_batch_queries(request: BatchQueryRequest):
    """Process multiple queries in batch."""
    try:
        if pipeline_manager.pipeline_agent is None:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        logger.info(f"Processing batch of {len(request.queries)} queries...")
        
        results = await pipeline_manager.batch_process_queries(
            queries=request.queries,
            max_concurrent=request.max_concurrent
        )
        
        return ApiResponse(
            success=True,
            data={
                "total_queries": len(request.queries),
                "results": results,
                "successful_queries": len([r for r in results if not r.get('error')]),
                "failed_queries": len([r for r in results if r.get('error')])
            },
            message=f"Batch processing completed for {len(request.queries)} queries"
        )
    except Exception as e:
        logger.error(f"Failed to process batch queries: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process batch queries: {str(e)}")

# @app.get("/results-history", response_model=ApiResponse)
# async def get_results_history(
#     limit: Optional[int] = None,
#     skip: Optional[int] = 0
# ):
#     """Get processing history with optional pagination."""
#     try:
#         history = pipeline_manager.results_history
        
#         # Apply pagination
#         if skip:
#             history = history[skip:]
#         if limit:
#             history = history[:limit]
        
#         return ApiResponse(
#             success=True,
#             data={
#                 "total_results": len(pipeline_manager.results_history),
#                 "returned_results": len(history),
#                 "results": history
#             },
#             message="Results history retrieved successfully"
#         )
#     except Exception as e:
#         logger.error(f"Failed to get results history: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to get results history: {str(e)}")

# @app.get("/summary-report", response_model=ApiResponse)
# async def get_summary_report():
#     """Get comprehensive summary report of all processed queries."""
#     try:
#         report = pipeline_manager.generate_summary_report()
        
#         return ApiResponse(
#             success=True,
#             data=report,
#             message="Summary report generated successfully"
#         )
#     except Exception as e:
#         logger.error(f"Failed to generate summary report: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to generate summary report: {str(e)}")

# @app.delete("/clear-history", response_model=ApiResponse)
# async def clear_results_history():
#     """Clear the results history."""
#     try:
#         pipeline_manager.results_history.clear()
        
#         return ApiResponse(
#             success=True,
#             data={"cleared_results": True},
#             message="Results history cleared successfully"
#         )
#     except Exception as e:
#         logger.error(f"Failed to clear results history: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to clear results history: {str(e)}")

# Background task endpoints
@app.post("/async-query", response_model=ApiResponse)
async def process_async_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks
):
    """Process a query asynchronously in the background."""
    try:
        if pipeline_manager.pipeline_agent is None:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        # Generate a task ID
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Add background task
        background_tasks.add_task(
            _process_background_query,
            task_id,
            request
        )
        
        return ApiResponse(
            success=True,
            data={"task_id": task_id, "status": "processing"},
            message="Query submitted for background processing"
        )
    except Exception as e:
        logger.error(f"Failed to submit async query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to submit async query: {str(e)}")

async def _process_background_query(task_id: str, request: QueryRequest):
    """Background task to process query."""
    try:
        logger.info(f"Background task {task_id} started")
        
        result = await pipeline_manager.pipeline_agent.process_comprehensive_query(
            user_query=request.user_query,
            local_model_id=request.local_model_id,
            context=request.context
        )
        
        # Add task_id to result
        result["task_id"] = task_id
        
        # Add to history
        pipeline_manager.results_history.append(result)
        
        logger.info(f"Background task {task_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Background task {task_id} failed: {str(e)}")
        
        # Add error result to history
        error_result = {
            "task_id": task_id,
            "user_query": request.user_query,
            "error": str(e),
            "processing_time": 0.0
        }
        pipeline_manager.results_history.append(error_result)

# Development server configuration
if __name__ == "__main__":
    # Configuration
    HOST = os.getenv("API_HOST", "127.0.0.1")
    PORT = int(os.getenv("API_PORT", 8009))
    DEBUG = os.getenv("DEBUG", "true").lower() == "true"
    
    logger.info(f"Starting Clinical Trials RAG Pipeline API on {HOST}:{PORT}")
    
    uvicorn.run(
        "api:app",
        host=HOST,
        port=PORT,
        reload=DEBUG,
        log_level="info" if DEBUG else "warning"
    )