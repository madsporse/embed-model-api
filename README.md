# Embed Model API

A production-ready FastAPI service for generating text embeddings using the multilingual-e5-large model. Supports both local model files and Hugging Face model loading with automatic device detection (CUDA/MPS/CPU).

## Repository Structure

```
embed-model-api/
├── .devcontainer/               # Development container configuration
│   ├── devcontainer.json
│   └── post-install.sh
├── app/                         # Main application package
│   ├── config.py                # Configuration settings with environment variables
│   ├── embeddings.py            # Model loading and embedding generation
│   ├── main.py                  # FastAPI application and endpoints
│   └── schemas.py               # Pydantic request/response models
├── models/                      # Local model storage (optional)
│   └── multilingual-e5-large/
├── scripts/                     # Utility scripts
│   ├── build-docker.sh
│   ├── download_model.py
│   └── run_tests.sh
├── tests/                       # Test suite
│   ├── conftest.py
│   ├── test_embed_unit.py
│   └── test_integration.py
├── Dockerfile                   # Container image definition
├── pyproject.toml               # Poetry configuration and dependencies
└── README.md                    # This file
```

## Getting Started

### Prerequisites

- Python 3.13+
- Poetry (for dependency management)

### Installation

#### 1. Using Poetry (Recommended)

Install Poetry if you haven't already:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Clone and setup the project:

```bash
git clone https://github.com/madsporse/embed-model-api
cd embed-model-api
poetry install
```

#### 2. Using GitHub Codespaces

1. Click "Code" → "Codespaces" → "Create codespace on main"
2. Wait for the environment to setup
3. Follow the "Running Locally" steps below

### Model Setup

You can use the model in two ways:

#### Option 1: Use Hugging Face Hub (Requires Internet)

By default, the code downloads the model directly from the Hugging Face Hub, so no additional setup is needed.

#### Option 2: Download Model Locally

If you prefer to download the model locally prior to running the API, you can do so with the following commands:

```bash
poetry run python scripts/download_model.py
```

Then, set the following environment variable prior to running the API:

```bash
export EMB_USE_LOCAL_MODEL="true"
```

### Configuration

The application uses environment variables with the `EMB_` prefix:

| Variable                 | Default                          | Description                       |
| ------------------------ | -------------------------------- | --------------------------------- |
| `EMB_MODEL_ID`           | `intfloat/multilingual-e5-large` | Hugging Face model ID             |
| `EMB_MAX_BATCH`          | `128`                            | Maximum batch size for requests   |
| `EMB_MAX_CHARS_PER_ITEM` | `8000`                           | Maximum characters per text item  |
| `EMB_DEFAULT_BATCH_SIZE` | `32`                             | Default batch size for processing |
| `EMB_CORS_ALLOW_ALL`     | `True`                           | Allow all CORS origins            |

## Running Locally

### Development Server

```bash
# Activate poetry environment
poetry shell

# Start the development server
poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Server

```bash
# Start with production settings
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

The API will be available at:

- **API**: http://localhost:8000
- **Interactive docs**: http://localhost:8000/docs
- **Health check**: http://localhost:8000/healthz

## Testing the API

### Using curl

```bash
# Single text embedding
curl -X POST "http://localhost:8000/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello world",
    "input_type": "passage",
    "normalize": true
  }'

# Multiple texts
curl -X POST "http://localhost:8000/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["Hello world", "Bonjour le monde"],
    "input_type": "passage",
    "normalize": true,
    "batch_size": 32
  }'
```

### Using Python

```python
import requests

response = requests.post("http://localhost:8000/embed", json={
    "input": ["Hello world", "How are you?"],
    "input_type": "passage",
    "normalize": True
})

data = response.json()
embeddings = [item["embedding"] for item in data["data"]]
print(f"Generated {len(embeddings)} embeddings of dimension {data['embedding_dimension']}")
```

### Running Tests

The project includes a unified test suite that can run with either fast mocked embedders (for development) or the real E5 model (for integration testing).

#### Test Options

```bash
# Fast tests with mocked embedder (recommended for development)
./scripts/run_tests.sh fast

# Full tests with real E5 model (slow, for integration)
./scripts/run_tests.sh real

# Only integration tests with real model
./scripts/run_tests.sh integration

# Docker container integration test
./scripts/run_tests.sh docker

# All tests with real model
./scripts/run_tests.sh all

# Tests with coverage report
./scripts/run_tests.sh coverage

# Tests with coverage report
./scripts/run_tests.sh coverage
```

#### Manual Test Commands

```bash
# Run fast tests with mocked embedder (default)
poetry run pytest tests/test_embed_unit.py -v

# Run tests with real E5 model
poetry run pytest tests/test_embed_unit.py --real-model -v

# Run only integration tests
poetry run pytest tests/test_integration.py -v

# Exclude integration tests (fast)
poetry run pytest tests/test_embed_unit.py -v

# Run tests with coverage
poetry run pytest --cov=app tests/ --cov-report=html
```

#### Test Structure

- **Mocked Embedder**: Fast, deterministic mocking of E5_Embedder for unit testing
- **Real Model**: Full E5 model for integration testing
- **Unified Tests**: Same test suite works with both mocked and real embedders
- **Mock Verification**: Tests can verify method calls, arguments, and inject errors
- **Command-Line Flag**: `--real-model` switches to real model testing

## API Endpoints

- **POST /embed**: Generate embeddings for text(s)
- **GET /healthz**: Basic health check
- **GET /readyz**: Readiness check
- **GET /docs**: Interactive API documentation
- **GET /metrics**: Prometheus metrics (if enabled)

### Usage Examples

#### Single Text Embedding

```bash
curl -X POST "http://localhost:8000/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, world!",
    "input_type": "passage",
    "normalize": true
  }'
```

#### Multiple Text Embeddings

```bash
curl -X POST "http://localhost:8000/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["Hello, world!", "How are you?", "This is a test."],
    "input_type": "passage",
    "normalize": true,
    "batch_size": 16
  }'
```

#### Query Embedding (for search/retrieval)

```bash
curl -X POST "http://localhost:8000/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "What is machine learning?",
    "input_type": "query",
    "normalize": true
  }'
```

**Response format:**

```json
{
  "model": "intfloat/multilingual-e5-large",
  "data": [
    {
      "index": 0,
      "embedding": [0.1, -0.2, 0.3, ...]
    }
  ],
  "embedding_dimension": 1024,
  "usage": {
    "total_input_tokens": 42
  }
}
```

## Device Support

The API automatically detects and uses the best available compute device:

1. **CUDA** (NVIDIA GPUs) - highest priority
2. **MPS** (Apple Silicon/ARM) - medium priority
3. **CPU** - fallback option
