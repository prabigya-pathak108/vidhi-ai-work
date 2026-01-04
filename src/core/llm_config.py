"""
LLM Configuration Loader
Loads LLM configurations from .env with flexible API key mapping
"""
import os
from typing import Any, Dict, Optional
from dataclasses import dataclass


@dataclass
class LLMConfig:
    """LLM Configuration"""
    provider: str
    model_name: str
    api_key: str
    temperature: float = 0.1


class LLMConfigLoader:
    """Load LLM configurations from environment variables"""
    
    def __init__(self, secrets_manager):
        """
        Initialize with secrets manager
        
        Args:
            secrets_manager: SecretManager instance
        """
        self.secrets = secrets_manager
    
    def _get_api_key(self, api_key_name: str) -> str:
        """
        Get API key by name from environment
        
        Args:
            api_key_name: Name of the env variable (e.g., "GROQ_API_KEY")
            
        Returns:
            API key value
            
        Raises:
            ValueError: If API key not found
        """
        api_key = self.secrets.get_from_env(api_key_name)
        if not api_key:
            raise ValueError(f"❌ {api_key_name} not found in .env file")
        return api_key
    
    def load_intent_llm_config(self) -> LLMConfig:
        """
        Load configuration for intent detection LLM
        
        Returns:
            LLMConfig object
        """
        provider = self.secrets.get_from_env("INTENT_LLM_PROVIDER")
        model_name = self.secrets.get_from_env("INTENT_LLM_MODEL")
        api_key_name = self.secrets.get_from_env("INTENT_LLM_API_KEY_NAME")
        temperature = float(self.secrets.get_from_env("INTENT_LLM_TEMPERATURE"))
        
        api_key = self._get_api_key(api_key_name)
        
        return LLMConfig(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            temperature=temperature
        )
    
    def load_legal_llm_config(self) -> LLMConfig:
        """
        Load configuration for legal question answering LLM
        
        Returns:
            LLMConfig object
        """
        provider = self.secrets.get_from_env("LEGAL_LLM_PROVIDER")
        model_name = self.secrets.get_from_env("LEGAL_LLM_MODEL")
        api_key_name = self.secrets.get_from_env("LEGAL_LLM_API_KEY_NAME")
        temperature = float(self.secrets.get_from_env("LEGAL_LLM_TEMPERATURE"))
        
        api_key = self._get_api_key(api_key_name)
        
        return LLMConfig(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            temperature=temperature
        )
    
    def load_embedding_config(self) -> Dict[str, Any]:
        """
        Load configuration for embedding model
        
        Returns:
            Dict with provider, model_name, api_key
        """
        provider = self.secrets.get_from_env("EMBEDDING_PROVIDER", default="pinecone")
        model_name = self.secrets.get_from_env("EMBEDDING_MODEL", default="multilingual-e5-large")
        api_key_name = self.secrets.get_from_env("EMBEDDING_API_KEY_NAME", default="PINECONE_API_KEY")
        
        api_key = self._get_api_key(api_key_name)
        
        return {
            "provider": provider,
            "model_name": model_name,
            "api_key": api_key
        }
    
    def load_vector_db_config(self) -> Dict[str, Any]:
        """
        Load configuration for vector database
        
        Returns:
            Dict with provider, api_key, index_name, cloud, region
        """
        provider = self.secrets.get_from_env("VECTOR_DB_PROVIDER", default="pinecone")
        api_key = self._get_api_key("PINECONE_API_KEY")
        index_name = self.secrets.get_from_env("PINECONE_INDEX_NAME", default="vidhi-ai-legal-index")
        cloud = self.secrets.get_from_env("PINECONE_CLOUD", default="aws")
        region = self.secrets.get_from_env("PINECONE_REGION", default="us-east-1")
        
        return {
            "provider": provider,
            "api_key": api_key,
            "index_name": index_name,
            "cloud": cloud,
            "region": region
        }
    
    def load_rag_config(self) -> Dict[str, Any]:
        """
        Load RAG configuration
        
        Returns:
            Dict with top_k, score_threshold
        """
        top_k = int(self.secrets.get_from_env("RAG_TOP_K", default="5"))
        score_threshold = float(self.secrets.get_from_env("RAG_SCORE_THRESHOLD", default="0.7"))
        
        return {
            "top_k": top_k,
            "score_threshold": score_threshold
        }
