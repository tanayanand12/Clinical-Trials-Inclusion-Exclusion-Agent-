FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY service_account_credentials.json ./ 

# Create necessary directories
RUN mkdir -p indexes csv-indexes gcp-indexes data temp_pdfs

# Expose port
EXPOSE 8000

# Start command
CMD ["python", "-m", "uvicorn", "src.enhanced_api_module:app", "--host", "0.0.0.0", "--port", "8000"]