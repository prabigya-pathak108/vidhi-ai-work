import os
import sys
from typing import Any, Dict, Optional, List
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError, ConfigDict
from langchain_core.output_parsers import PydanticOutputParser
import json
import time
from src.utils.logger import get_logger

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.prompt_factory import PromptFactory
from src.core.vector_factory import VectorDBBase

logger = get_logger(__name__)


# ===== PYDANTIC MODELS =====
class IntentResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "language": "roman-nepali",
                "is_follow_up": False,
                "enhanced_question": None,
                "intent": "legal_query",
                "cleaned_question": "What is the property law in Nepal?"
            }
        }
    )
    
    language: str = Field(description="Language: 'english', 'nepali', or 'roman-nepali'")
    is_follow_up: bool = Field(description="Is this a follow-up question?")
    enhanced_question: Optional[str] = Field(default=None, description="Enhanced question if follow-up")
    intent: str = Field(description="Intent: 'legal_query', 'greeting', 'off_topic'")
    cleaned_question: str = Field(description="Cleaned question in English")


class LegalAnswerResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "answer": "According to Nepal's Civil Code...",
                "confidence": "high",
                "sources_used": ["Civil Code 2074"],
                "disclaimer": "This is general legal information..."
            }
        }
    )
    
    answer: str = Field(description="The legal answer")
    confidence: str = Field(description="Confidence: 'high', 'medium', or 'low'")
    sources_used: List[str] = Field(default_factory=list, description="Legal sources referenced")
    disclaimer: Optional[str] = Field(default=None, description="Legal disclaimer")


# ===== SIMPLE LLM SERVICE =====
class LLMService:
    def __init__(
        self, 
        intent_model: Any, 
        legal_model: Any,
        embed_model: Any,
        vector_db: VectorDBBase,
        index_name: str,
        top_k: int = 5
    ):
        """
        Simple service with vector search integration
        
        Args:
            intent_model: LLM for intent detection
            legal_model: LLM for answering legal questions
            embed_model: Embedding model for vector search
            vector_db: Vector database instance (any provider)
            index_name: Name of the vector index/collection
            top_k: Default number of documents to retrieve
        """
        self.intent_model = intent_model
        self.legal_model = legal_model
        self.embed_model = embed_model
        self.vector_db = vector_db 
        self.index_name = index_name
        self.top_k = top_k
        
        self.intent_parser = PydanticOutputParser(pydantic_object=IntentResponse)
        self.legal_parser = PydanticOutputParser(pydantic_object=LegalAnswerResponse)
        
        logger.info("✅ LLMService initialized with vector database integration")

    def _clean_json(self, raw_output: str) -> str:
        """Remove markdown code blocks from JSON"""
        output = raw_output.strip()
        if output.startswith("```json"):
            output = output[7:]
        elif output.startswith("```"):
            output = output[3:]
        if output.endswith("```"):
            output = output[:-3]
        return output.strip()

    def identify_intent(self, user_query: str, conversation_history: List[str]) -> IntentResponse:
        """Identify user intent with retry logic"""
        logger.debug(f"🎯 Identifying intent for query: {user_query[:50]}...")
        
        # Don't use format_instructions - it's hardcoded in the template now
        prompt = PromptFactory.get_prompt(
            "identify_intent",
            user_question=user_query,
            conversation_history="\n".join(conversation_history) if conversation_history else "No previous conversation"
        )
        
        # Try 3 times
        for attempt in range(3):
            try:
                response = self.intent_model.invoke(prompt)
                cleaned = self._clean_json(response.content)
                result = self.intent_parser.parse(cleaned)
                logger.info(f"✅ Intent identified: {result.intent} (language: {result.language})")
                return result
            except Exception as e:
                logger.warning(f"⚠️ Intent detection attempt {attempt + 1} failed: {str(e)[:100]}")
                if attempt == 2:  # Last attempt
                    logger.error(f"❌ Intent detection failed after 3 attempts")
                    raise
                time.sleep(1)

    def retrieve_legal_context(self, question: str, top_k: int = 5) -> tuple[str, List[Dict]]:
        """
        Retrieve relevant legal documents from vector database
        
        Args:
            question: User's question
            top_k: Number of documents to retrieve
            
        Returns:
            tuple: (combined_context_string, list_of_source_documents)
        """
        logger.info(f"🔍 Searching vector database for: '{question[:50]}...'")
        
        # Get embedding for the question
        query_embedding = self.embed_model.embed_query(question)
        logger.debug(f"📊 Generated query embedding (dimension: {len(query_embedding)})")
        
        # Search vector database (works with any provider)
        results = self.vector_db.search(
            collection_name=self.index_name,
            query_vector=query_embedding,
            top_k=top_k
        )
        
        if not results:
            logger.warning("⚠️ No relevant documents found in vector database")
            return "No relevant legal documents found.", []
        
        logger.info(f"✅ Found {len(results)} relevant documents")
        
        # Combine all retrieved content for LLM context
        context_parts = []
        # Store actual sources for frontend display
        sources_list = []
        
        for idx, result in enumerate(results, 1):
            # Get content from metadata
            content = result.metadata.get('content', result.metadata.get('text', ''))
            act = result.metadata.get('act', 'Unknown')
            section = result.metadata.get('section', '')
            chapter = result.metadata.get('chapter', '')
            
            logger.debug(f"  Document {idx}: {act} - Section {section} (Score: {result.score:.3f})")
            
            # Format for LLM context
            context_parts.append(
                f"[Document {idx}] (Score: {result.score:.3f})\n"
                f"Source: {act}, Section {section}\n"
                f"Content: {content}\n"
            )
            
            # Store source information for frontend
            source_info = {
                'act': act,
                'section': section,
                'chapter': chapter,
                'content': content[:500],  # Limit content length
                'score': round(result.score, 3)
            }
            sources_list.append(source_info)
        
        combined_context = "\n".join(context_parts)
        return combined_context, sources_list

    def answer_legal_question(
        self, 
        question: str,
        intented_question_result: str,
        conversation_history: List[str] = None,
        retrieve_context: bool = True,
        context: str = None,
        top_k: int = None
    ) -> LegalAnswerResponse:
        """
        Answer legal question with context retrieval
        
        Args:
            question: The legal question
            intented_question_result: Result from intent detection
            conversation_history: Previous conversation history
            retrieve_context: Whether to retrieve context from vector DB
            context: Manual context (if retrieve_context=False)
            top_k: Number of documents to retrieve (uses default if None)
            
        Returns:
            LegalAnswerResponse: Structured answer
        """
        logger.info(f"🤖 Generating legal answer for: {question[:50]}...")
        
        # Use instance default if not specified
        if top_k is None:
            top_k = self.top_k
        
        # Get context from vector DB if needed
        if retrieve_context:
            context = self.retrieve_legal_context(question, top_k=top_k)[0]
        elif context is None:
            context = "No context provided."
        
        # Generate answer
        format_instructions = self.legal_parser.get_format_instructions()
        prompt = PromptFactory.get_prompt(
            "answer_legal_question",
            question=question,
            intented_question_result=intented_question_result,
            conversation_history="\n".join(conversation_history) if conversation_history else "",
            context=context,
            format_instructions=format_instructions
        )
        
        # Try 3 times
        for attempt in range(3):
            try:
                response = self.legal_model.invoke(prompt)
                cleaned = self._clean_json(response.content)
                result = self.legal_parser.parse(cleaned)
                logger.info(f"✅ Legal answer generated (confidence: {result.confidence})")
                return result
            except Exception as e:
                logger.warning(f"⚠️ Answer generation attempt {attempt + 1} failed: {str(e)[:100]}")
                if attempt == 2:
                    logger.error(f"❌ Answer generation failed after 3 attempts")
                    raise
                time.sleep(1)

    def process_query(
        self, 
        user_query: str, 
        conversation_history: List[str] = None
    ) -> Dict[str, Any]:
        """
        Main function: Process user query end-to-end
        
        Args:
            user_query: User's question
            conversation_history: Previous conversation
            
        Returns:
            Dict with intent, answer, and actual retrieved sources
        """
        if conversation_history is None:
            conversation_history = []
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📝 Processing query: {user_query}")
        logger.info(f"{'='*60}")
        
        # Step 1: Identify intent
        try:
            intent = self.identify_intent(user_query, conversation_history)
            logger.info(f"🎯 Intent: {intent.intent}, Language: {intent.language}")
            
            result = {
                "intent": intent.model_dump(),
                "answer": None,
                "sources_used": []
            }
            
            # Step 2: If legal query, retrieve context and answer
            if intent.intent == "legal_query":
                logger.info(f"📚 Legal query detected - processing with RAG...")
                
                # Retrieve context AND actual sources
                context, retrieved_sources = self.retrieve_legal_context(
                    question=user_query,
                    top_k=5
                )
                
                # Generate answer using the context
                format_instructions = self.legal_parser.get_format_instructions()
                prompt = PromptFactory.get_prompt(
                    "answer_legal_question",
                    question=user_query,
                    intented_question_result=str(intent),
                    conversation_history="\n".join(conversation_history) if conversation_history else "",
                    context=context,
                    format_instructions=format_instructions
                )
                
                # Try 3 times to get answer
                for attempt in range(3):
                    try:
                        response = self.legal_model.invoke(prompt)
                        cleaned = self._clean_json(response.content)
                        answer = self.legal_parser.parse(cleaned)
                        break
                    except Exception as e:
                        logger.warning(f"⚠️ Attempt {attempt + 1} failed: {str(e)[:100]}")
                        if attempt == 2:
                            raise
                        time.sleep(1)
                
                # Return answer with ACTUAL retrieved sources
                result = {
                    "answer": answer.answer,
                    "confidence": answer.confidence,
                    "sources_used": retrieved_sources,  # Use actual retrieved docs
                    "disclaimer": answer.disclaimer
                }
                logger.info(f"✅ Legal answer generated successfully")
            
            elif intent.intent == "greeting":
                logger.info("👋 Greeting detected")
                result = {
                    "answer": "नमस्कार! How can I help you with Nepali legal information?",
                    "confidence": "high",
                    "sources_used": [],
                    "disclaimer": None
                }
            
            else:
                logger.info(f"⚠️ Off-topic query detected")
                result = {
                    "answer": "Sorry me, I can only help with questions about Nepali law. Please ask me a legal question.",
                    "confidence": "high",
                    "sources_used": [],
                    "disclaimer": None
                }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing query: {e}", exc_info=True)
            return {
                "error": str(e),
                "intent": {"intent": "error"},
                "answer": None,
                "sources_used": []
            }
