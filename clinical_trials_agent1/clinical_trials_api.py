from fastapi import FastAPI, HTTPException #type: ignore
from fastapi.middleware.cors import CORSMiddleware #type: ignore
from pydantic import BaseModel, Field #type: ignore
from typing import Optional, Dict, Any
import logging
import json
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv #type: ignore
from openai import OpenAI #type: ignore
import asyncio
import uvicorn #type: ignore

# Load environment variables
load_dotenv()

# Configure logging with UTF-8 encoding
def setup_logging():
    log_dir = Path("api_logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create file handler with UTF-8 encoding
    file_handler = logging.FileHandler(
        log_dir / "api.log", 
        encoding='utf-8'
    )
    
    # Create console handler with UTF-8 encoding
    console_handler = logging.StreamHandler()
    
    # Set the same formatter for both handlers
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Prevent duplicate logs
    logger.propagate = False
    
    return logger

logger = setup_logging()

# Import the pipeline class (assuming it's available)
try:
    from .clinical_trials_rag_pipeline import ClinicalTrialsRAGPipeline
    PIPELINE_AVAILABLE = True
except ImportError:
    logger.warning("ClinicalTrialsRAGPipeline not available. Using mock responses.")
    PIPELINE_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(
    title="Clinical Trials Research API",
    description="API for querying clinical trials data using RAG pipeline",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline = None

# Pydantic models
class QueryRequest(BaseModel):
    query: str = Field(..., description="Research question to process")
    max_trials: Optional[int] = Field(20, description="Maximum number of trials to analyze")
    top_k: Optional[int] = Field(10, description="Number of top relevant chunks to use")

class QueryResponse(BaseModel):
    success: bool
    answer: Optional[str] = None
    studies_count: int = 0
    processing_time: float = 0.0
    quality_score: float = 0.0
    error: Optional[str] = None
    studies: Optional[list] = None
    metadata: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    pipeline_initialized: bool
    timestamp: str

# Mock pipeline for testing when actual pipeline is unavailable
class MockPipeline:
    def __init__(self):
        self.initialized = True
    
    def process_query(self, query: str, top_k: int = 10):
        import time
        import random
        
        # Simulate processing time
        time.sleep(random.uniform(0.5, 2.0))
        
        return {
            'success': True,
            'answer': f"Mock analysis for query: '{query}'. This is a simulated response showing how the API would work with actual clinical trials data. The system would analyze relevant studies and provide evidence-based insights.",
            'studies': [
                {
                    'study_id': f'NCT{random.randint(10000000, 99999999)}',
                    'title': f'Mock Study {i+1} for Clinical Research',
                    'similarity_score': random.uniform(0.7, 0.95)
                } for i in range(random.randint(3, 8))
            ],
            'metadata': {
                'processing_time': random.uniform(0.5, 2.0),
                'total_trials_fetched': random.randint(50, 200),
                'trials_processed': random.randint(10, 30),
                'chunks_created': random.randint(100, 500),
                'relevant_chunks': top_k
            },
            'quality_assessment': {
                'overall_score': random.uniform(0.6, 0.95),
                'quality_level': 'High' if random.random() > 0.3 else 'Medium'
            }
        }
    
    def get_pipeline_status(self):
        return {
            'pipeline_initialized': True,
            'components': {
                'openai_client': True,
                'embedding_model': True,
                'vector_store': True,
                'query_processor': True
            }
        }

# Initialize pipeline on startup using lifespan
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await startup_event()
    yield
    # Shutdown (if needed)

app = FastAPI(
    title="Clinical Trials Research API",
    description="API for querying clinical trials data using RAG pipeline",
    version="1.0.0",
    lifespan=lifespan
)

async def startup_event():
    global pipeline
    logger.info("Initializing Clinical Trials Research API...")
    
    try:
        if PIPELINE_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            pipeline = ClinicalTrialsRAGPipeline(
                openai_client=openai_client,
                model_name="gpt-4-turbo",
                embedding_model="text-embedding-ada-002",
                max_trials=25,
                max_context_length=8000,
                chunk_size=1000,
                chunk_overlap=200
            )
            logger.info("Pipeline initialized successfully with OpenAI")
        else:
            pipeline = MockPipeline()
            logger.warning("Using mock pipeline - check OPENAI_API_KEY and pipeline availability")
            
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        pipeline = MockPipeline()
        logger.info("Fallback to mock pipeline")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API and pipeline health status"""
    try:
        status = pipeline.get_pipeline_status() if pipeline else {"pipeline_initialized": False}
        return HealthResponse(
            status="healthy",
            pipeline_initialized=status.get("pipeline_initialized", False),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

# Main query processing endpoint
@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a clinical trials research query"""
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    if not request.query or len(request.query.strip()) < 5:
        raise HTTPException(status_code=400, detail="Query must be at least 5 characters long")
    
    logger.info(f"Processing query: {request.query[:100]}...")
    start_time = datetime.now()
    
    try:
        # Process the query
        result = pipeline.process_query(
            query=request.query,
            top_k=request.top_k
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        if result['success']:
            logger.info(f"Query processed successfully in {processing_time:.2f}s")
            
            return QueryResponse(
                success=True,
                answer=result['answer'],
                studies_count=len(result['studies']),
                processing_time=result['metadata']['processing_time'],
                quality_score=result['quality_assessment']['overall_score'],
                studies=result['studies'][:10],  # Limit studies in response
                metadata=result['metadata']
            )
        else:
            logger.error(f"Query processing failed: {result['error']}")
            return QueryResponse(
                success=False,
                error=result['error'],
                processing_time=processing_time
            )
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing query: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {error_msg}")

# Get pipeline status
@app.get("/status")
async def get_pipeline_status():
    """Get detailed pipeline status"""
    if not pipeline:
        return {"error": "Pipeline not initialized"}
    
    try:
        return pipeline.get_pipeline_status()
    except Exception as e:
        logger.error(f"Failed to get pipeline status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get pipeline status")

# Example queries endpoint
@app.get("/examples")
async def get_example_queries():
    """Get example research queries"""
    examples = [
        {
            "query": "What are the success rates of transpedal access compared to femoral access?",
            "category": "Comparative Effectiveness",
            "description": "Comparing different access methods"
        },
        {
            "query": "How many studies compare different sheath sizes for arterial access?",
            "category": "Equipment Analysis",
            "description": "Medical device comparisons"
        },
        {
            "query": "What are the reported complication rates in peripheral interventions?",
            "category": "Safety Analysis",
            "description": "Adverse event assessment"
        },
        {
            "query": "Which closure techniques show the lowest complication rates?",
            "category": "Procedural Techniques",
            "description": "Technique effectiveness comparison"
        },
        {
            "query": "What patient populations are most commonly studied in vascular trials?",
            "category": "Demographics",
            "description": "Patient characteristics analysis"
        }
    ]
    return {"examples": examples}

# Root endpoint
@app.get("/")
async def root():
    """API welcome message"""
    return {
        "message": "Clinical Trials Research API",
        "version": "1.0.0",
        "endpoints": {
            "POST /query": "Process research queries",
            "GET /health": "Health check",
            "GET /status": "Pipeline status",
            "GET /examples": "Example queries",
            "GET /docs": "API documentation"
        }
    }

# Run the API
if __name__ == "__main__":
    uvicorn.run(
        "clinical_trials_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )