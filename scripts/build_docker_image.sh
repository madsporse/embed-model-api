#!/bin/bash
set -e

# Change to the project root directory (parent of scripts/)
cd "$(dirname "$0")/.."

echo "Building Docker image for Embedding API..."

# Check if models directory exists
if [ ! -d "models/multilingual-e5-large" ]; then
    echo "Local model not found. Downloading model first..."
    poetry run python scripts/download_model.py
fi

# Build the Docker image
docker build -t embedding-api:latest .

echo "Docker image built successfully!"
echo ""
echo "To run the container:"
echo "  docker run -p 8000:8000 embedding-api:latest"
echo ""
echo "To run in background:"
echo "  docker run -d -p 8000:8000 --name embedding-api embedding-api:latest"
echo ""
echo "To use Hugging Face model instead of local:"
echo "  docker run -p 8000:8000 -e EMB_USE_LOCAL_MODEL=false embedding-api:latest"
