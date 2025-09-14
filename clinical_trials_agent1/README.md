# Clinical Trials RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system for querying and analyzing clinical trials data from ClinicalTrials.gov. This system fetches relevant clinical trial data based on user queries, processes it intelligently, and provides detailed, evidence-based answers.

## 🚀 Features

- **Intelligent Data Fetching**: Uses progressive search strategies to find relevant clinical trials
- **Smart Chunking**: Semantically chunks clinical trial data for optimal retrieval
- **Vector Search**: OpenAI-powered embeddings for finding most relevant information
- **Comprehensive Analysis**: Generates detailed answers with proper citations and study references
- **Quality Assessment**: Built-in response quality validation
- **Fast Execution**: Optimized for low latency and efficient processing

## 📋 Requirements

Ensure you have the following dependencies installed (from requirements.txt):

```
openai~=1.73.0
python-dotenv==1.0.1
numpy==1.26.3
requests>=2.31.0
typing-extensions>=4.7.0
```

## 🔧 Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**:
   Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## 🏗️ System Architecture

The system consists of several modular components:

### Core Components

1. **ClinicalTrialsFetcherAgent** (`fetcher.py`)
   - Fetches clinical trial data from ClinicalTrials.gov API
   - Implements progressive search strategies for empty results
   - Handles various query types and filters

2. **ClinicalTrialsChunker** (`clinical_trials_chunker.py`)
   - Intelligently chunks clinical trial data into semantic segments
   - Extracts key sections: overview, interventions, outcomes, eligibility, etc.
   - Handles large content with overlap strategies

3. **ClinicalTrialsVectorizer** (`clinical_trials_vectorizer.py`)
   - Vectorizes chunks using OpenAI embeddings (text-embedding-ada-002)
   - Batch processing with rate limiting
   - Computes similarity scores for retrieval

4. **ClinicalTrialsContextExtractor** (`clinical_trials_context_extractor.py`)
   - Extracts most relevant chunks based on query similarity
   - Prioritizes chunk types based on query intent
   - Formats context within token limits

5. **ClinicalTrialsRAGModule** (`clinical_trials_rag_module.py`)
   - Generates comprehensive answers using OpenAI GPT models
   - Specialized prompts for clinical trial analysis
   - Quality validation and response assessment

6. **ClinicalTrialsRAGPipeline** (`clinical_trials_rag_pipeline.py`)
   - Main orchestrator that connects all components
   - End-to-end query processing
   - Performance monitoring and error handling

## 🚀 Quick Start

### Basic Usage

```python
from clinical_trials_rag_pipeline import ClinicalTrialsRAGPipeline
from openai import OpenAI
import os

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize pipeline
pipeline = ClinicalTrialsRAGPipeline(
    openai_client=openai_client,
    model_name="gpt-4-turbo",
    max_trials=20
)

# Process a query
result = pipeline.process_query(
    "How many recruiting clinical trials are there for Type 2 Diabetes in India?"
)

if result['success']:
    print(f"Answer: {result['answer']}")
    print(f"Studies analyzed: {len(result['studies'])}")
else:
    print(f"Error: {result['error']}")
```

### Advanced Configuration

```python
pipeline = ClinicalTrialsRAGPipeline(
    openai_client=openai_client,
    model_name="gpt-4-turbo",           # or "gpt-3.5-turbo" for speed
    embedding_model="text-embedding-ada-002",
    max_trials=25,                      # Max trials to fetch
    max_context_length=8000,            # Max context for LLM
    chunk_size=1000,                    # Size of each chunk
    chunk_overlap=200,                  # Overlap between chunks
)
```

## 🎯 Example Queries

The system handles various types of clinical trial queries:

### Count/Statistical Queries
```python
"How many recruiting clinical trials are there for Type 2 Diabetes in India?"
"How many completed studies exist for cardiovascular disease?"
```

### Intervention/Treatment Queries
```python
"What interventions are being tested for diabetes treatment?"
"What drugs are being studied for Alzheimer's disease?"
```

### Eligibility Criteria Queries
```python
"What are the common eligibility criteria for diabetes trials?"
"What age groups are included in cancer studies?"
```

### Location-Based Queries
```python
"Which hospitals in India are conducting diabetes research?"
"Where are COVID-19 vaccine trials being conducted?"
```

### Outcome Measures Queries
```python
"What are the primary endpoints in diabetes trials?"
"What outcomes are measured in heart disease studies?"
```

## 📊 Response Structure

The system returns comprehensive results:

```python
{
    "query": "User's original question",
    "success": True,
    "answer": "Detailed analysis with citations",
    "citations": ["List of study citations"],
    "studies": [
        {
            "study_id": "NCT12345678",
            "title": "Study title",
            "similarity_score": 0.85
        }
    ],
    "metadata": {
        "processing_time": 3.45,
        "total_trials_fetched": 15,
        "chunks_created": 75,
        "relevant_chunks": 8,
        "model_used": "gpt-4-turbo"
    },
    "quality_assessment": {
        "overall_score": 0.92,
        "quality_level": "high"
    }
}
```

## 🧪 Running Examples

### Demonstration Script
```bash
python example_usage.py
```

### Interactive Mode
```bash
python example_usage.py interactive
```

## ⚙️ Configuration Options

### Pipeline Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | "gpt-4-turbo" | OpenAI model for answer generation |
| `embedding_model` | "text-embedding-ada-002" | OpenAI model for embeddings |
| `max_trials` | 20 | Maximum trials to fetch and process |
| `max_context_length` | 8000 | Maximum context length for LLM |
| `chunk_size` | 1000 | Size of each chunk in characters |
| `chunk_overlap` | 200 | Overlap between chunks |

### Context Extraction Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_similarity_threshold` | 0.3 | Minimum similarity to include chunk |
| `top_k` | 10 | Number of top chunks to retrieve |

## 🔍 How It Works

1. **Query Analysis**: The system analyzes the user query to understand intent and extract key terms

2. **Data Fetching**: Uses `ClinicalTrialsFetcherAgent` to fetch relevant trials from ClinicalTrials.gov with progressive search strategies

3. **Data Processing**: Chunks the fetched trial data into semantic segments focusing on:
   - Study overview and design
   - Interventions and treatments
   - Outcomes and endpoints
   - Eligibility criteria
   - Location and sponsor information

4. **Vectorization**: Converts chunks to embeddings using OpenAI's text-embedding-ada-002

5. **Retrieval**: Finds most relevant chunks based on query similarity and content type prioritization

6. **Answer Generation**: Uses GPT-4 to generate comprehensive, structured answers with proper citations

7. **Quality Assessment**: Validates response quality and provides metrics

## 🚨 Error Handling

The system includes comprehensive error handling:

- **API Failures**: Graceful degradation with retry mechanisms
- **Empty Results**: Progressive search strategies to find relevant data
- **Token Limits**: Automatic truncation and chunking
- **Rate Limiting**: Built-in delays and backoff strategies

## 📈 Performance Tips

1. **Model Selection**: Use "gpt-3.5-turbo" for faster responses, "gpt-4-turbo" for higher quality
2. **Chunk Size**: Smaller chunks (800-1000) provide better granularity
3. **Context Length**: Adjust based on model limits and response requirements
4. **Caching**: Consider implementing caching for repeated queries

## 🔒 Security Notes

- Store API keys securely in environment variables
- Never commit API keys to version control
- Consider rate limiting for production use
- Validate user inputs to prevent injection attacks

## 📝 Logging

The system uses Python's logging module. Configure logging level:

```python
import logging
logging.basicConfig(level=logging.INFO)  # or DEBUG for detailed logs
```

## 🤝 Contributing

To extend the system:

1. **Add New Data Sources**: Extend `ClinicalTrialsFetcherAgent` to support additional APIs
2. **Improve Chunking**: Enhance `ClinicalTrialsChunker` for better semantic segmentation
3. **Custom Prompts**: Modify `ClinicalTrialsRAGModule` for specialized use cases
4. **Performance Optimization**: Add caching layers or async processing

## 📚 API Reference

### ClinicalTrialsRAGPipeline

Main class for end-to-end processing:

- `process_query(query: str, top_k: int = 10) -> Dict[str, Any]`
- `get_pipeline_status() -> Dict[str, Any]`

### Individual Components

Each component can be used independently for specialized use cases. See individual module documentation for detailed API references.

## ⚠️ Limitations

- Dependent on ClinicalTrials.gov API availability
- Limited by OpenAI API rate limits and costs
- Context length limited by model token limits
- May not capture all nuances in very complex medical terminology

## 🆘 Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure OPENAI_API_KEY is set correctly
2. **Empty Results**: System uses progressive search; check query specificity
3. **Slow Performance**: Reduce max_trials or chunk_size for faster processing
4. **Rate Limits**: Implement delays between requests if hitting limits

### Debug Mode

Enable debug logging for detailed information:

```python
logging.basicConfig(level=logging.DEBUG)
```

## 📞 Support

For issues or questions:
1. Check the error logs for specific error messages
2. Verify API key configuration
3. Test with simpler queries first
4. Review the example usage patterns

This system provides a robust foundation for clinical trials research and can be extended for specialized medical research applications.