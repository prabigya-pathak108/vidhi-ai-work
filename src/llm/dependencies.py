"""
LLM Dependencies for FastAPI
Provides singleton instances of LLM services
"""
import os
import sys
from functools import lru_cache
from src.utils.logger import get_logger

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.secrets import SecretManager
from src.core.llm_factory import LLMFactory
from src.core.embed_factory import EmbeddingFactory
from src.core.vector_factory import VectorStoreFactory
from src.core.llm_config import LLMConfigLoader
from src.llm.calls import LLMService

logger = get_logger(__name__)


@lru_cache()
def get_llm_service() -> LLMService:
    """
    Get singleton instance of LLMService with configuration from .env
    
    Returns:
        LLMService instance
    """
    logger.info("🔧 Initializing LLM Service from .env configuration...")
    
    # Load configurations
    secrets = SecretManager()
    config_loader = LLMConfigLoader(secrets)
    
    # Load all configs
    logger.debug("📋 Loading configuration from environment...")
    intent_config = config_loader.load_intent_llm_config()
    legal_config = config_loader.load_legal_llm_config()
    embed_config = config_loader.load_embedding_config()
    vector_config = config_loader.load_vector_db_config()
    rag_config = config_loader.load_rag_config()
    
    logger.info(f"✅ Intent LLM: {intent_config.provider}/{intent_config.model_name}")
    logger.info(f"✅ Legal LLM: {legal_config.provider}/{legal_config.model_name}")
    logger.info(f"✅ Embedding: {embed_config['provider']}/{embed_config['model_name']}")
    logger.info(f"✅ Vector DB: {vector_config['provider']}/{vector_config['index_name']}")
    logger.info(f"✅ RAG Config: top_k={rag_config['top_k']}")
    
    # Initialize models
    logger.debug("🔨 Creating intent model...")
    intent_model = LLMFactory.create_llm(
        provider=intent_config.provider,
        model_name=intent_config.model_name,
        api_key=intent_config.api_key,
        temperature=intent_config.temperature
    )
    
    logger.debug("🔨 Creating legal model...")
    legal_model = LLMFactory.create_llm(
        provider=legal_config.provider,
        model_name=legal_config.model_name,
        api_key=legal_config.api_key,
        temperature=legal_config.temperature
    )
    
    logger.debug("🔨 Creating embedding model...")
    embed_model = EmbeddingFactory.create_embedding_model(
        provider=embed_config['provider'],
        api_key=embed_config['api_key'],
        model_name=embed_config['model_name']
    )
    
    logger.debug("🔨 Connecting to vector database...")
    vector_db = VectorStoreFactory.create(
        provider=vector_config['provider'],
        api_key=vector_config['api_key'],
        cloud=vector_config['cloud'],
        region=vector_config['region']
    )
    
    # Create service
    logger.debug("🔨 Assembling LLM service...")
    service = LLMService(
        intent_model=intent_model,
        legal_model=legal_model,
        embed_model=embed_model,
        vector_db=vector_db,
        index_name=vector_config['index_name'],
        top_k=rag_config['top_k']
    )
    
    logger.info("✅ LLM Service initialized successfully!")
    return service
