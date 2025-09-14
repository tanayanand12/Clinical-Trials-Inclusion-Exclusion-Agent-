# Clinical Trials RAG Library Integration Guide

## Package Structure

```
clinical_trials_rag/
├── __init__.py                          # Main package initialization
├── clinical_trials_rag_pipeline.py      # Main pipeline
├── fetcher.py                           # Data fetching
├── clinical_trials_chunker.py           # Text chunking
├── clinical_trials_vectorizer.py        # Embeddings
├── clinical_trials_context_extractor.py # Context extraction
├── clinical_trials_rag_module.py        # RAG generation
├── clinical_trials_api.py               # Optional API server
└── config.py                            # Configuration
```

## Installation

### From Source
```bash
git clone https://github.com/yourusername/clinical-trials-rag.git
cd clinical-trials-rag
pip install -e .
```

### As a Package
```bash
pip install clinical-trials-rag
```

## Basic Usage

```python
from clinical_trials_rag import ClinicalTrialsRAG

# Initialize
rag = ClinicalTrialsRAG(api_key="your-openai-api-key")

# Query
result = rag.query("How many diabetes studies are recruiting?")

# Access results
print(result.answer)
print(f"Quality score: {result.quality_score}")
print(f"Studies found: {len(result.studies)}")
```

## Integration Examples

### 1. Django Integration

```python
# views.py
from django.http import JsonResponse
from clinical_trials_rag import ClinicalTrialsRAG

# Initialize once in settings or as singleton
rag = ClinicalTrialsRAG()

def search_trials(request):
    query = request.GET.get('query')
    result = rag.query(query)
    
    return JsonResponse({
        'answer': result.answer,
        'studies': result.studies,
        'quality_score': result.quality_score
    })
```

### 2. Flask Integration

```python
from flask import Flask, request, jsonify
from clinical_trials_rag import ClinicalTrialsRAG

app = Flask(__name__)
rag = ClinicalTrialsRAG()

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    result = rag.query(data['query'])
    return jsonify(result.to_dict())
```

### 3. Async Integration

```python
import asyncio
from clinical_trials_rag import ClinicalTrialsRAG

async def process_queries(queries):
    rag = ClinicalTrialsRAG()
    
    tasks = []
    for query in queries:
        task = asyncio.create_task(
            asyncio.to_thread(rag.query, query)
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results
```

### 4. Batch Processing

```python
from clinical_trials_rag import ClinicalTrialsRAG
import pandas as pd

def batch_process_queries(queries_df):
    rag = ClinicalTrialsRAG(max_trials=30)
    
    results = []
    for _, row in queries_df.iterrows():
        result = rag.query(row['question'])
        results.append({
            'question': row['question'],
            'answer': result.answer,
            'quality_score': result.quality_score,
            'studies_count': len(result.studies)
        })
    
    return pd.DataFrame(results)
```

## Advanced Configuration

```python
from clinical_trials_rag import ClinicalTrialsRAG

# Custom configuration
rag = ClinicalTrialsRAG(
    api_key="your-key",
    model_name="gpt-4-turbo",
    embedding_model="text-embedding-ada-002",
    max_trials=50,              # More trials for comprehensive analysis
    chunk_size=1000,            # Larger chunks
    chunk_overlap=200,          # More overlap
    max_context_length=8000     # Longer context
)
```

## Error Handling

```python
from clinical_trials_rag import ClinicalTrialsRAG

try:
    rag = ClinicalTrialsRAG()
    result = rag.query("Your question here")
    
    if result.success:
        print(result.answer)
    else:
        print(f"Query failed: {result.error}")
        
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Using Individual Components

```python
from clinical_trials_rag import (
    ClinicalTrialsFetcherAgent,
    ClinicalTrialsChunker,
    ClinicalTrialsVectorizer
)

# Use components separately
fetcher = ClinicalTrialsFetcherAgent(openai_client)
chunker = ClinicalTrialsChunker(chunk_size=800)
vectorizer = ClinicalTrialsVectorizer()

# Custom pipeline
data = fetcher.analyze_user_query("diabetes studies")
chunks = chunker.chunk_clinical_trials_data(data['data'])
embeddings = vectorizer.embed_chunks(chunks)
```

## Environment Variables

Create a `.env` file:
```
OPENAI_API_KEY=your-api-key-here
```

Or set programmatically:
```python
import os
os.environ['OPENAI_API_KEY'] = 'your-api-key'
```

## Logging Configuration

```python
import logging
from clinical_trials_rag import ClinicalTrialsRAG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Disable verbose logging
logging.getLogger('clinical_trials_rag').setLevel(logging.WARNING)
```

## Testing Your Integration

```python
import unittest
from clinical_trials_rag import ClinicalTrialsRAG

class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.rag = ClinicalTrialsRAG()
    
    def test_basic_query(self):
        result = self.rag.query("diabetes studies")
        self.assertTrue(result.success)
        self.assertIsNotNone(result.answer)
        self.assertGreater(len(result.studies), 0)
```

## Performance Considerations

1. **Initialize Once**: Create the RAG instance once and reuse it
2. **Batch Queries**: Process multiple queries together when possible
3. **Cache Results**: Consider caching frequent queries
4. **Adjust Parameters**: Tune `max_trials` and `chunk_size` for your use case

## Common Issues

1. **API Key**: Ensure OPENAI_API_KEY is set
2. **Memory**: Large queries may require more memory
3. **Rate Limits**: Add delays between queries if needed
4. **Timeouts**: Increase timeout for complex queries