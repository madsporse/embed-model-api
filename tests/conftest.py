import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app, get_embedder


def pytest_addoption(parser):
    """Add custom command-line options for pytest."""
    parser.addoption(
        "--real-model",
        action="store_true",
        default=False,
        help="Use real E5 model instead of mocks (slower but more realistic)"
    )


@pytest.fixture(scope="session")
def use_real_model(request):
    """Session-scoped fixture that determines if real model should be used."""
    return request.config.getoption("--real-model")


@pytest.fixture(autouse=True)
def mock_embedder(use_real_model):
    """Mock the E5_Embedder class and override dependency injection.

    Uses a simple MagicMock with minimal, predictable behavior so individual
    tests can override return values or side effects as needed. Avoids
    embedding-specific hashing/logic in the test harness.
    """
    if use_real_model:
        # When using real model, set up dependency override with real embedder
        from app.embeddings import E5_Embedder
        real_embedder = E5_Embedder()
        app.dependency_overrides[get_embedder] = lambda: real_embedder
        yield
        app.dependency_overrides.clear()
        return

    # Patch the E5_Embedder class and provide a MagicMock instance.
    with patch('app.embeddings.E5_Embedder') as mock_embedder_class:
        mock_instance = MagicMock()
        mock_instance.dim = 8
        mock_embedder_class.return_value = mock_instance
        app.dependency_overrides[get_embedder] = lambda: mock_instance
        mock_instance._mock_class = mock_embedder_class
        yield mock_instance
        app.dependency_overrides.clear()


@pytest.fixture
def client():
    """Provide a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def embedder_instance(mock_embedder, use_real_model):
    """Provide the embedder instance for direct testing of embedder methods.
    
    Use this fixture when you want to test the embedder directly (not via API).
    This returns the same mock instance that the FastAPI app uses.
    """
    if use_real_model:
        from app.embeddings import E5_Embedder
        return E5_Embedder()
    else:
        return mock_embedder


@pytest.fixture  
def configured_mock(mock_embedder, use_real_model):
    """Provide a pre-configured mock for API tests.
    
    Use this fixture when you need to configure mock behavior for API tests.
    Returns the same mock instance that the FastAPI app uses.
    """
    if use_real_model:
        # When using real model, return None to indicate no mock configuration needed
        return None
    else:
        # Set up default mock behavior that works for most API tests
        import numpy as np
        mock_embedder.dim = 8
        mock_embedder.encode.return_value = np.ones((1, 8), dtype=np.float32)
        mock_embedder.tokenize_count.return_value = 10
        return mock_embedder


@pytest.fixture
def is_real_model(use_real_model):
    """Boolean fixture indicating if we're testing with real model."""
    return use_real_model