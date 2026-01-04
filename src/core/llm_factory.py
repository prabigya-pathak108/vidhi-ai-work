from typing import Any
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from src.utils.logger import get_logger

logger = get_logger(__name__)

class LLMFactory:
    @staticmethod
    def create_llm(
        provider: str,
        model_name: str,
        api_key: str,
        temperature: float = 0.7,
        **kwargs: Any
    ) -> BaseChatModel:
        
        logger.debug(f"🔨 Creating LLM: provider={provider}, model={model_name}, temp={temperature}")
        
        if not api_key:
            logger.error(f"❌ API key is required for {provider}")
            raise ValueError(f"API key is required for {provider}")

        provider = provider.lower().strip()

        if provider == "openai":
            logger.info(f"✅ Created OpenAI LLM: {model_name}")
            return ChatOpenAI(
                model=model_name,
                api_key=api_key,
                temperature=temperature,
                **kwargs
            )

        elif provider == "anthropic":
            logger.info(f"✅ Created Anthropic LLM: {model_name}")
            return ChatAnthropic(
                model_name=model_name,
                api_key=api_key,
                temperature=temperature,
                **kwargs
            )
        elif provider == "google" or provider == "gemini":
            logger.info(f"✅ Created Google Gemini LLM: {model_name}")
            return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=temperature, **kwargs)

        elif provider == "groq":
            logger.info(f"✅ Created Groq LLM: {model_name}")
            return ChatGroq(
                model_name=model_name,
                api_key=api_key,
                temperature=temperature,
                **kwargs
            )

        elif provider == "huggingface":
            logger.info(f"✅ Created HuggingFace LLM: {model_name}")
            llm = HuggingFaceEndpoint(
                repo_id=model_name,
                huggingfacehub_api_token=api_key,
                temperature=temperature,
                **kwargs
            )
            return ChatHuggingFace(llm=llm)

        else:
            logger.error(f"❌ Unsupported provider: {provider}")
            raise ValueError(f"Unsupported provider: {provider}")

if __name__ == "__main__":
    import sys
    import os
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.core.secrets import SecretManager

    secrets = SecretManager()
    logger.info("Starting Vidhi-AI LLM Factory Test")

    # A. TEST GROQ
    logger.info("Testing Groq...")
    groq_key = secrets.get_from_env("GROQ_API_KEY")
    if groq_key:
        try:
            groq_llm = LLMFactory.create_llm(
                provider="groq",
                model_name="llama-3.3-70b-versatile",
                api_key=groq_key,
                temperature=0.1
            )
            response = groq_llm.invoke("In one sentence, what is a 'Dapha' in Nepal Law?")
            logger.info(f"Groq Response: {response.content}")
        except Exception as e:
            logger.error(f"Groq Error: {e}")
    else:
        logger.info("Skipping Groq (No Key)")

    # B. TEST GEMINI
    logger.info("Testing Gemini...")
    google_key = secrets.get_from_env("GEMINI_API_KEY")
    if google_key:
        try:
            gemini_llm = LLMFactory.create_llm(
                provider="google",
                model_name="gemini-2.5-flash",
                api_key=google_key,
                temperature=0.1
            )
            # Testing with Nepali context
            response = gemini_llm.invoke("नेपालको कानुनमा 'दफा' भनेको के हो? छोटो उत्तर दिनुहोस्।")
            logger.info(f"Gemini Response: {response.content}")
        except Exception as e:
            logger.error(f"Gemini Error: {e}")
    else:
        logger.info("Skipping Gemini (No Key)")