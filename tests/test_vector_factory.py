"""
Test Vector Store Factory - validates vector database creation
"""
import pytest
from src.core.vector_factory import VectorStoreFactory


def test_vector_factory_requires_api_key():
    """Test that vector factory requires API key for Pinecone"""
    with pytest.raises(ValueError, match="Pinecone requires api_key"):
        VectorStoreFactory.create(
            provider="pinecone",
            api_key=None
        )


def test_vector_factory_rejects_invalid_provider():
    """Test that invalid provider raises error"""
    with pytest.raises(ValueError, match="Unsupported provider"):
        VectorStoreFactory.create(
            provider="invalid_provider",
            api_key="test-key"
        )
