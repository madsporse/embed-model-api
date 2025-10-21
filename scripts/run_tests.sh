#!/bin/bash
# Test runner script for the embedding API
# This script provides convenient ways to run different test configurations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    echo "Embedding API Test Runner"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  fast                 Run fast tests with mocked embedder (default)"
    echo "  real                 Run tests with real E5 model (slow, uses --real-model flag)"
    echo "  integration          Run only integration tests with real model"
    echo "  docker               Run only Docker container integration test"
    echo "  all                  Run all tests with real model"
    echo "  coverage             Run tests with coverage (mocked embedder)"
    echo "  help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 fast              # Quick test run for development"
    echo "  $0 real              # Full test with real model"
    echo "  $0 integration       # Only integration tests"
    echo "  $0 docker            # Only Docker container test"
    echo ""
}

# Set PYTHONPATH
export PYTHONPATH=$(pwd)

case "${1:-fast}" in
    "fast")
        print_status "Running fast tests with mocked embedder..."
        poetry run pytest tests/test_embed_unit.py -v --durations=10
        ;;
    
    "real")
        print_status "Running tests with real E5 model (this will be slow)..."
        print_warning "This will load the real E5 model and may take several minutes"
        poetry run pytest tests/test_embed_unit.py --real-model -v --durations=10
        ;;
    
    "integration")
        print_status "Running only integration tests with real E5 model..."
        poetry run pytest tests/test_integration.py -v --durations=10
        ;;
    
    "docker")
        print_status "Running Docker container integration test..."
        print_warning "This will build a Docker image and may take several minutes"
        # Use the integration docker container test; this will build and run a container
        poetry run pytest tests/test_integration.py -k test_container_smoke -v --durations=10
        ;;
    
    "all")
        print_status "Running ALL tests with real E5 model..."
        print_warning "This will run the full test suite and may take a long time"
        poetry run pytest --real-model -v --durations=10
        ;;
    
    "coverage")
        print_status "Running tests with coverage report (mocked embedder)..."
        poetry run pytest tests/test_embed_unit.py --cov=app --cov-report=html --cov-report=term -v
        print_status "Coverage report generated in htmlcov/"
        ;;
    
    "help"|"-h"|"--help")
        show_help
        ;;
    
    *)
        print_error "Unknown option: $1"
        echo ""
        show_help
        exit 1
        ;;
esac

print_status "Tests completed!"