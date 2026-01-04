"""
Test Embedding Factory - validates embedding model creation
"""
import pytest
from src.core.embed_factory import EmbeddingFactory


def test_embedding_factory_requires_model_name():
    """Test that embedding factory requires model name"""
    with pytest.raises(ValueError, match="Model name is required"):
        EmbeddingFactory.create_embedding_model(
            provider="pinecone",
            api_key="test-key",
            model_name=None
        )


def test_embedding_factory_rejects_invalid_provider():
    """Test that invalid provider raises error"""
    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        EmbeddingFactory.create_embedding_model(
            provider="invalid_provider",
            api_key="test-key",
            model_name="model"
        )
