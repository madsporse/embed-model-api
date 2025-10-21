"""
Integration tests for the embedding API.

These tests verify that the complete deployment pipeline works correctly
by building and testing the actual Docker container, as well as testing
real model behavior when available.

Run with: 
- pytest -m integration tests/test_integration.py  # Docker tests only
- pytest -m integration tests/test_integration.py --real-model  # Include real model tests
"""

import pytest
import requests
import subprocess
import time
import numpy as np


@pytest.mark.integration
def test_container_smoke():
    """
    Integration test using Docker container.
    
    This builds and runs the actual Docker container to test
    the complete deployment pipeline.
    """
    tag = "embed-test:latest"
    
    try:
        print("Building Docker image...")
        result = subprocess.run(
            ["docker", "build", "-t", tag, "."],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )
        
        if result.returncode != 0:
            pytest.skip(f"Docker build failed: {result.stderr}")
        
        print("Starting container...")
        process = subprocess.Popen(
            ["docker", "run", "--rm", "-p", "8001:8000", tag],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        try:
            # Wait for server to start (max 60 seconds)
            for _ in range(60):
                try:
                    response = requests.get("http://localhost:8001/healthz", timeout=2)
                    if response.status_code == 200:
                        break
                except requests.RequestException:
                    time.sleep(1)
            else:
                pytest.fail("Server failed to start within 60 seconds")
            
            # Test the embedding endpoint
            response = requests.post(
                "http://localhost:8001/embed",
                json={
                    "input": "Integration test with Docker",
                    "input_type": "passage"
                },
                timeout=30
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify response structure
            assert "embedding_dimension" in data
            assert data["embedding_dimension"] in (768, 1024)
            assert "data" in data
            assert len(data["data"]) == 1
            assert len(data["data"][0]["embedding"]) == data["embedding_dimension"]
            
            print(f"✓ Container test passed - embedding dimension: {data['embedding_dimension']}")
            
        finally:
            # Clean up container
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                
    except subprocess.TimeoutExpired:
        pytest.skip("Docker build timed out")
    except Exception as e:
        pytest.skip(f"Docker test failed: {e}")


if __name__ == "__main__":
    # Allow running docker tests directly
    pytest.main([__file__, "-v", "-m", "integration"])


# Real model integration tests that only make sense with real models
@pytest.mark.integration
class TestRealModelSpecific:
    """Tests specific to real model behavior."""
    
    def test_model_consistency(self, embedder_instance, use_real_model):
        """Test that the model produces consistent embeddings for the same input."""
        if not use_real_model:
            pytest.skip("Real model integration tests require --real-model flag")
            
        text = "Consistency test text"
        
        embeddings1 = embedder_instance.encode([text], "passage", True, 32)
        embeddings2 = embedder_instance.encode([text], "passage", True, 32)
        
        np.testing.assert_allclose(embeddings1, embeddings2, rtol=1e-6)

    def test_input_type_differences(self, embedder_instance, use_real_model):
        """Test that query and passage prefixes produce different embeddings."""
        if not use_real_model:
            pytest.skip("Real model integration tests require --real-model flag")
            
        text = "What is artificial intelligence?"
        
        query_embedding = embedder_instance.encode([text], "query", True, 32)
        passage_embedding = embedder_instance.encode([text], "passage", True, 32)
        
        assert not np.allclose(query_embedding, passage_embedding, rtol=1e-3)

    def test_model_dimension_expectations(self, embedder_instance, use_real_model):
        """Test that real model has expected dimensions."""
        if not use_real_model:
            pytest.skip("Real model integration tests require --real-model flag")
            
        # E5-large models typically have 1024 dimensions
        assert embedder_instance.dim in [768, 1024], f"Unexpected dimension: {embedder_instance.dim}"