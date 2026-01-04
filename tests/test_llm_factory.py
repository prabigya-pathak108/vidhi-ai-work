"""
Test LLM Factory - validates provider creation
"""
import pytest
from src.core.llm_factory import LLMFactory


def test_llm_factory_requires_api_key():
    """Test that LLM factory rejects empty API key"""
    with pytest.raises(ValueError, match="API key is required"):
        LLMFactory.create_llm(
            provider="groq",
            model_name="llama-3.3-70b-versatile",
            api_key="",
            temperature=0.7
        )


def test_llm_factory_rejects_invalid_provider():
    """Test that invalid provider raises error"""
    with pytest.raises(ValueError, match="Unsupported provider"):
        LLMFactory.create_llm(
            provider="invalid_provider",
            model_name="model",
            api_key="test-key"
        )
