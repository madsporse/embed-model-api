"""
Unit tests for the embedding API.

This module contains comprehensive tests for the embedding API functionality, including
HTTP endpoint tests, validation tests, and mock behavior verification.

Run with mocked embedder (fast): pytest tests/test_embed_unit.py
Run with real model (slow): pytest tests/test_embed_unit.py --real-model
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient

import pytest
import os
from fastapi.testclient import TestClient


class TestEmbedAPI:
    """Tests for the embedding API endpoints using HTTP requests."""
    
    def test_embed_single_text(self, client: TestClient, configured_mock):
        """Test embedding a single text."""
        if configured_mock:
            # Configure specific mock behavior for this test
            configured_mock.encode.return_value = np.arange(1, 9, dtype=np.float32).reshape(1, -1)
            configured_mock.tokenize_count.return_value = 42

        response = client.post(
            "/embed",
            json={
                "input": "Hello world",
                "input_type": "passage"
            }
        )
        assert response.status_code == 200

        data = response.json()
        assert "embedding_dimension" in data
        assert len(data["data"]) == 1
        assert data["data"][0]["index"] == 0
        assert isinstance(data["data"][0]["embedding"], list)
        assert "model" in data
        assert "usage" in data
        if configured_mock:
            assert data["usage"]["total_input_tokens"] == 42


    def test_embed_multiple_texts(self, client: TestClient, configured_mock):
        """Test embedding multiple texts."""
        texts = ["Hello world", "Bonjour", "Hola", "Ciao", "Hallo"]
        
        if configured_mock:
            mock_embedding = np.tile(np.arange(1, 9, dtype=np.float32), (len(texts), 1))
            configured_mock.encode.return_value = mock_embedding
            configured_mock.tokenize_count.return_value = 100
            
        response = client.post(
            "/embed",
            json={
                "input": texts,
                "input_type": "passage"
            }
        )
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["data"]) == len(texts)
        assert "embedding_dimension" in data
        assert "model" in data
        assert "usage" in data
        
        # Verify each embedding item has correct structure
        for i, item in enumerate(data["data"]):
            assert "index" in item
            assert "embedding" in item
            assert item["index"] == i
            assert isinstance(item["embedding"], list)
            assert len(item["embedding"]) == data["embedding_dimension"]

    @pytest.mark.parametrize("input_type", ["passage", "query"])
    def test_embed_input_types(self, client: TestClient, configured_mock, input_type: str):
        """Test different input types (passage vs query)."""
        if configured_mock:
            configured_mock.encode.return_value = np.arange(1, 9, dtype=np.float32).reshape(1, -1)
            configured_mock.tokenize_count.return_value = 10
            
        response = client.post(
            "/embed",
            json={
                "input": "What is artificial intelligence?",
                "input_type": input_type
            }
        )
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["data"]) == 1
        assert isinstance(data["data"][0]["embedding"], list)
        assert len(data["data"][0]["embedding"]) == data["embedding_dimension"]

    @pytest.mark.parametrize("normalize", [True, False])
    def test_embed_normalization(self, client: TestClient, configured_mock, normalize: bool):
        """Test embedding normalization options."""
        if configured_mock:
            if normalize:
                # Return a normalized vector (L2 norm = 1)
                mock_embedding = np.ones((1, 8), dtype=np.float32) / np.sqrt(8)
            else:
                mock_embedding = np.arange(1, 9, dtype=np.float32).reshape(1, -1)
            configured_mock.encode.return_value = mock_embedding
            configured_mock.tokenize_count.return_value = 8
            
        response = client.post(
            "/embed",
            json={
                "input": "Test normalization",
                "input_type": "passage",
                "normalize": normalize
            }
        )
        assert response.status_code == 200
        
        data = response.json()
        embedding = data["data"][0]["embedding"]
        
        if normalize and configured_mock:
            # Check if embedding is approximately normalized (L2 norm ≈ 1)
            import math
            norm = math.sqrt(sum(x*x for x in embedding))
            assert abs(norm - 1.0) < 0.1  # Allow some tolerance
        else:
            # For non-normalized, we just check it's a valid embedding
            assert len(embedding) == data["embedding_dimension"]

    def test_embed_batch_processing(self, client: TestClient, configured_mock):
        """Test batch processing with custom batch size."""
        texts = [f"Text {i}" for i in range(5)]
        
        if configured_mock:
            mock_embedding = np.tile(np.arange(1, 9, dtype=np.float32), (len(texts), 1))
            configured_mock.encode.return_value = mock_embedding
            configured_mock.tokenize_count.return_value = 50
            
        response = client.post(
            "/embed",
            json={
                "input": texts,
                "input_type": "passage",
                "batch_size": 2
            }
        )
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["data"]) == 5
        # All embeddings should be processed regardless of batch size
        for item in data["data"]:
            assert len(item["embedding"]) == data["embedding_dimension"]

    def test_response_structure(self, client: TestClient, configured_mock):
        """Test that response has all required fields with correct types."""
        if configured_mock:
            configured_mock.encode.return_value = np.arange(1, 9, dtype=np.float32).reshape(1, -1)
            configured_mock.tokenize_count.return_value = 8
            
        response = client.post(
            "/embed",
            json={
                "input": "Test response structure",
                "input_type": "passage"
            }
        )
        assert response.status_code == 200
        
        data = response.json()
        
        # Check top-level structure
        required_fields = ["model", "data", "embedding_dimension", "usage"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Check data structure
        assert isinstance(data["data"], list)
        assert len(data["data"]) == 1
        
        item = data["data"][0]
        assert "index" in item
        assert "embedding" in item
        assert isinstance(item["index"], int)
        assert isinstance(item["embedding"], list)
        
        # Check usage structure
        assert "total_input_tokens" in data["usage"]
        assert isinstance(data["usage"]["total_input_tokens"], int)
        assert data["usage"]["total_input_tokens"] > 0


class TestEmbedValidation:
    """Test input validation and error handling."""
    
    def test_empty_input_validation(self, client: TestClient, embedder_instance):
        """Test validation with empty input."""
        # No need to mock embedder since validation fails before calling it
        response = client.post(
            "/embed",
            json={
                "input": [],
                "input_type": "passage"
            }
        )
        # Should return validation error (422) due to empty list
        assert response.status_code == 422

    def test_empty_string_validation(self, client: TestClient, embedder_instance):
        """Test validation with empty string."""
        # No need to mock embedder since validation fails before calling it
        response = client.post(
            "/embed",
            json={
                "input": "",
                "input_type": "passage"
            }
        )
        # Should return validation error due to empty string
        assert response.status_code == 422

    def test_validation_max_chars_per_item(self, client: TestClient, monkeypatch):
        """Test character limit validation per item."""
        from app.config import settings
        monkeypatch.setattr(settings, "MAX_CHARS_PER_ITEM", 10)
        
        response = client.post(
            "/embed",
            json={
                "input": "This text is definitely longer than 10 characters",
                "input_type": "passage"
            }
        )
        assert response.status_code == 413
        assert "exceeds" in response.json()["detail"]

    def test_validation_max_batch_size(self, client: TestClient, monkeypatch):
        """Test batch size limit validation."""
        from app.config import settings
        monkeypatch.setattr(settings, "MAX_BATCH", 2)
        
        response = client.post(
            "/embed",
            json={
                "input": ["Text 1", "Text 2", "Text 3"],
                "input_type": "passage"
            }
        )
        assert response.status_code == 413
        assert "Max batch size" in response.json()["detail"]

    def test_invalid_input_type(self, client: TestClient):
        """Test validation with non-string input."""
        response = client.post(
            "/embed",
            json={
                "input": [123, "valid string"],  # Mixed types
                "input_type": "passage"
            }
        )
        assert response.status_code == 422  # Pydantic validation error
        error_detail = response.json()["detail"]
        # Pydantic returns list of validation errors
        assert isinstance(error_detail, list)
        # Check that it mentions string validation
        error_str = str(error_detail)
        assert "string" in error_str.lower()

    def test_invalid_input_type_enum(self, client: TestClient):
        """Test validation with invalid input_type enum value."""
        response = client.post(
            "/embed",
            json={
                "input": "Test text",
                "input_type": "invalid_type"  # Not in InputType enum
            }
        )
        assert response.status_code == 422  # Pydantic validation error

    def test_batch_size_boundary_conditions(self, client: TestClient, configured_mock):
        """Test batch_size parameter boundary conditions."""
        if configured_mock:
            configured_mock.encode.return_value = np.arange(1, 9, dtype=np.float32).reshape(1, -1)
            configured_mock.tokenize_count.return_value = 5
        
        # Test batch_size = 1 (minimum valid)
        response = client.post(
            "/embed",
            json={
                "input": "Test text",
                "input_type": "passage",
                "batch_size": 1
            }
        )
        assert response.status_code == 200
        
        # Test batch_size = 0 (invalid)
        response = client.post(
            "/embed",
            json={
                "input": "Test text", 
                "input_type": "passage",
                "batch_size": 0
            }
        )
        assert response.status_code == 422  # Should fail validation


class TestEmbedderDirect:
    """Direct tests of the embedder instance (bypassing API)."""
    
    def test_embedder_properties(self, embedder_instance, is_real_model):
        """Test basic embedder properties."""
        if not is_real_model:
            embedder_instance.dim = 8
        assert hasattr(embedder_instance, 'dim')
        assert embedder_instance.dim > 0
        
        # For mock embedder, we know it's 8; for real embedder it should be larger
        if is_real_model:
            assert embedder_instance.dim >= 768  # Real models typically have large dimensions
        else:
            assert embedder_instance.dim == 8  # Mock embedder dimension

    def test_embedder_encode(self, embedder_instance, is_real_model):
        """Test direct encoding functionality."""
        texts = ["Hello", "World"]
        embed_dim = 8
        import numpy as np
        
        if is_real_model:
            # Test with real model - just verify it works
            embeddings = embedder_instance.encode(texts, "passage", True, 32)
            assert embeddings.shape[0] == len(texts)
            assert embeddings.shape[1] == embedder_instance.dim  # Real model has its own dimension
            norms = np.linalg.norm(embeddings, axis=1)
            np.testing.assert_allclose(norms, 1.0, rtol=1e-2)  # Allow some tolerance for real model
        else:
            # Test with mock - configure and verify behavior
            mock_embedding = np.ones((len(texts), embed_dim), dtype=np.float32) / np.sqrt(embed_dim)
            embedder_instance.encode.return_value = mock_embedding
            embeddings = embedder_instance.encode(texts, "passage", True, 32)
            assert embeddings.shape[0] == len(texts)
            assert embeddings.shape[1] == embed_dim
            norms = np.linalg.norm(embeddings, axis=1)
            np.testing.assert_allclose(norms, 1.0, rtol=1e-5)

    def test_embedder_tokenize_count(self, embedder_instance, is_real_model):
        """Test token counting functionality."""
        texts = ["Hello world", "This is a test"]
        
        if is_real_model:
            # Test with real model - just verify it works
            token_count = embedder_instance.tokenize_count(texts, "passage")
            assert isinstance(token_count, int)
            assert token_count > 0
            
            longer_texts = texts + ["Additional text for counting"]
            longer_count = embedder_instance.tokenize_count(longer_texts, "passage")
            assert isinstance(longer_count, int)
            assert longer_count > token_count  # More text should have more tokens
        else:
            # Test with mock - configure and verify behavior
            # First call returns 10, second call returns 15
            embedder_instance.tokenize_count.side_effect = [10, 15]
            
            token_count = embedder_instance.tokenize_count(texts, "passage")
            assert isinstance(token_count, int)
            assert token_count > 0
            assert token_count == 10
            
            longer_texts = texts + ["Additional text for counting"]
            longer_count = embedder_instance.tokenize_count(longer_texts, "passage")
            assert isinstance(longer_count, int)
            assert longer_count > token_count
            assert longer_count == 15


class TestMockBehavior:
    """Tests specific to mock embedder behavior and call verification."""
    
    def test_mock_embedder_calls(self, embedder_instance, client: TestClient, is_real_model):
        """Test that mock embedder methods are called with correct arguments."""
        if is_real_model:
            pytest.skip("Mock-specific tests only run with mock embedder")
            
        # Reset call history
        embedder_instance.encode.reset_mock()
        embedder_instance.tokenize_count.reset_mock()
        
        # Make API call
        response = client.post(
            "/embed",
            json={
                "input": ["Test text 1", "Test text 2"],
                "input_type": "passage",
                "normalize": True,
                "batch_size": 16
            }
        )
        
        assert response.status_code == 200
        
        # Verify encode was called with correct arguments
        embedder_instance.encode.assert_called_once_with(
            ["Test text 1", "Test text 2"],
            "passage",
            True,
            16
        )
        
        # Verify tokenize_count was called with correct arguments
        embedder_instance.tokenize_count.assert_called_once_with(
            ["Test text 1", "Test text 2"],
            "passage"
        )

    def test_mock_deterministic_output(self, embedder_instance, is_real_model):
        """Test that mock embedder provides deterministic output."""
        if is_real_model:
            pytest.skip("Mock-specific tests only run with mock embedder")
            
        texts = ["Same text", "Another text"]
        
        # Call encode multiple times
        embeddings1 = embedder_instance.encode(texts, "passage", True, 32)
        embeddings2 = embedder_instance.encode(texts, "passage", True, 32)
        
        # Should be identical (deterministic)
        import numpy as np
        np.testing.assert_array_equal(embeddings1, embeddings2)

    def test_mock_call_count(self, embedder_instance, client: TestClient, is_real_model):
        """Test that we can verify call counts on mock embedder."""
        if is_real_model:
            pytest.skip("Mock-specific tests only run with mock embedder")
            
        # Reset call history
        embedder_instance.encode.reset_mock()
        embedder_instance.tokenize_count.reset_mock()
        
        # Make multiple API calls
        for i in range(3):
            client.post(
                "/embed",
                json={
                    "input": f"Test text {i}",
                    "input_type": "passage"
                }
            )
        
        # Verify methods were called 3 times each
        assert embedder_instance.encode.call_count == 3
        assert embedder_instance.tokenize_count.call_count == 3

    def test_mock_error_injection(self, embedder_instance, client: TestClient, is_real_model):
        """Test that we can inject errors through the mock for error handling tests."""
        if is_real_model:
            pytest.skip("Mock-specific tests only run with mock embedder")
            
        # Configure mock to raise an exception
        embedder_instance.encode.side_effect = RuntimeError("Mock embedding error")
        
        response = client.post(
            "/embed",
            json={
                "input": "Test text",
                "input_type": "passage"
            }
        )
        
        # Should return 500 error due to mock exception
        assert response.status_code == 500
        assert "Mock embedding error" in response.json()["detail"]


class TestHealthEndpoints:
    """Test health and readiness endpoints."""
    
    def test_health_endpoint(self, client: TestClient):
        """Test the health check endpoint."""
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_readiness_endpoint(self, client: TestClient):
        """Test the readiness check endpoint."""
        response = client.get("/readyz")
        assert response.status_code == 200
        assert response.json() == {"status": "ready"}


class TestOpenAPISchema:
    """Test OpenAPI schema generation."""
    
    def test_openapi_schema(self, client: TestClient):
        """Test that OpenAPI schema is generated correctly."""
        response = client.get("/docs")
        assert response.status_code == 200
        
        # Test that we can get the OpenAPI JSON
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        
        # Basic schema validation
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        assert "/embed" in schema["paths"]
        assert "post" in schema["paths"]["/embed"]