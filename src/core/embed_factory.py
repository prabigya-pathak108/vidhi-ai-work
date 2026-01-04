from typing import Any, Optional
from langchain_core.embeddings import Embeddings
from src.utils.logger import get_logger

logger = get_logger(__name__)

class EmbeddingFactory:
    """
    Factory to create embedding model instances.
    Returns a LangChain 'Embeddings' compatible object.
    """
    
    @staticmethod
    def create_embedding_model(
        provider: str,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs: Any
    ) -> Embeddings:
        """
        Args:
            provider: 'openai', 'google', 'huggingface', 'pinecone'
            api_key: API Key for the provider
            model_name: Specific model version
        """
        
        logger.debug(f"🔨 Creating embedding model: provider={provider}, model={model_name}")
        provider = provider.lower().strip()

        # ------------------------------------------------
        # 1. Pinecone Inference (NEW)
        # ------------------------------------------------
        if provider == "pinecone":
            from langchain_pinecone import PineconeEmbeddings
            
            if not api_key:
                logger.error("❌ Pinecone API Key required for inference")
                raise ValueError("Pinecone API Key required for inference.")
            
            if not model_name:
                logger.error("❌ Model name is required for Pinecone inference")
                raise ValueError("Model name is required for Pinecone inference.")

            model = model_name.strip()
            logger.info(f"✅ Created Pinecone embedding model: {model}")
            
            return PineconeEmbeddings(
                model=model,
                pinecone_api_key=api_key,
                **kwargs
            )

        elif provider == "openai":
            from langchain_openai import OpenAIEmbeddings

            if not model_name:
                logger.error("❌ Model name is required for OpenAI inference")
                raise ValueError("Model name is required for OpenAI inference.")

            model = model_name.strip()
            
            if not api_key:
                logger.error("❌ OpenAI API Key required")
                raise ValueError("OpenAI API Key required.")

            logger.info(f"✅ Created OpenAI embedding model: {model}")
            return OpenAIEmbeddings(
                model=model,
                api_key=api_key,
                **kwargs
            )

        elif provider == "google" or provider == "gemini":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            if not model_name:
                logger.error("❌ Model name is required for Gemini inference")
                raise ValueError("Model name is required for Gemini inference.")

            model = model_name.strip()
            
            if not api_key:
                logger.error("❌ Google API Key required")
                raise ValueError("Google API Key required.")

            logger.info(f"✅ Created Google embedding model: {model}")
            return GoogleGenerativeAIEmbeddings(
                model=model,
                google_api_key=api_key,
                **kwargs
            )
        elif provider == "huggingface":
            from langchain_huggingface import HuggingFaceEndpointEmbeddings
            if not model_name:
                logger.error("❌ Model name is required for HuggingFace inference")
                raise ValueError("Model name is required for HuggingFace inference.")

            model = model_name.strip()
            
            if not api_key:
                # Fallback to Local execution
                logger.warning("⚠️ No API Key for HuggingFace. Falling back to Local execution.")
                from langchain_huggingface import HuggingFaceEmbeddings
                return HuggingFaceEmbeddings(model_name=model, **kwargs)
            
            logger.info(f"✅ Created HuggingFace embedding model: {model}")
            return HuggingFaceEndpointEmbeddings(
                model=model,
                huggingfacehub_api_token=api_key,
                **kwargs
            )

        else:
            logger.error(f"❌ Unsupported embedding provider: {provider}")
            raise ValueError(f"Unsupported embedding provider: {provider}")

if __name__ == "__main__":
    import sys
    import os

    # 1. DYNAMIC PATH FIX
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # 2. Imports after path fix
    from src.core.secrets import SecretManager

    secrets = SecretManager()
    logger.info("Starting Vidhi-AI Embedding Test")

    api_key = secrets.get_from_env("PINECONE_API_KEY")

    if not api_key:
        logger.error("PINECONE_API_KEY not found in environment")
    else:
        try:
            # 1. Create Pinecone Embeddings
            logger.info("Initializing Pinecone Embeddings...")
            embed_model = EmbeddingFactory.create_embedding_model(
                provider="pinecone",
                api_key=api_key,
                model_name="multilingual-e5-large"
            )

            # 2. Test Embedding
            sentences = ["Hello world", "नमस्ते, तपाईँलाई कस्तो छ?"]
            batch_vecs = embed_model.embed_documents(sentences)
            
            logger.info(f"Success! Dimension: {len(batch_vecs[0])}, Vectors: {len(batch_vecs)}")
            logger.debug(f"Sample: {batch_vecs[0][:3]}...")

        except Exception as e:
            logger.error(f"Execution Error: {e}")