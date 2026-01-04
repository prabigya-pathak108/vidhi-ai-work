from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from src.database.session import get_db
from src.database import models
from src.schemas import chat_schemas
from src.crud import chat_ops
from src.llm.dependencies import get_llm_service
from src.llm.calls import LLMService
import uuid
from typing import List
from src.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/session/{user_id}", status_code=status.HTTP_201_CREATED)
def start_session(user_id: str, db: Session = Depends(get_db)):
    """
    Create a new chat session for a user
    
    Args:
        user_id: User identifier
        db: Database session
    
    Returns:
        Session ID and metadata
    """
    logger.info(f"📝 Session creation request for user: {user_id}")
    
    # Verify User exists
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        logger.warning(f"❌ User not found: {user_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"User with id '{user_id}' not found. Please create user first."
        )
    
    try:
        new_session = chat_ops.create_chat_session(db, user_id)
        logger.info(f"✅ Created session {new_session.id} for user {user_id}")
        
        return {
            "session_id": new_session.id,
            "user_id": user_id,
            "created_at": new_session.created_at,
            "message": "Session created successfully"
        }
    except Exception as e:
        logger.error(f"❌ Failed to create session: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create chat session"
        )


@router.get("/session/{session_id}", response_model=chat_schemas.SessionInfo)
def get_session_info(session_id: uuid.UUID, db: Session = Depends(get_db)):
    """
    Get session information with message count
    
    Args:
        session_id: Chat session UUID
        db: Database session
    
    Returns:
        Session metadata including message count
    """
    logger.info(f"📊 Session info request: {session_id}")
    session_data = chat_ops.get_session_with_count(db, session_id)
    
    if not session_data:
        logger.warning(f"❌ Session not found: {session_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    
    logger.info(f"✅ Session info retrieved: {session_id}")
    return chat_schemas.SessionInfo(
        session_id=session_data["session"].id,
        user_id=session_data["session"].user_id,
        created_at=session_data["session"].created_at,
        message_count=session_data["message_count"]
    )


@router.get("/history/{session_id}", response_model=List[chat_schemas.MessageResponse])
def get_history(session_id: uuid.UUID, db: Session = Depends(get_db)):
    """
    Get conversation history for a session
    
    Args:
        session_id: Chat session UUID
        db: Database session
    
    Returns:
        List of messages in chronological order
    """
    logger.info(f"📜 History request for session: {session_id}")
    
    # Verify session exists
    session = db.query(models.ChatSession).filter(
        models.ChatSession.id == session_id
    ).first()
    
    if not session:
        logger.warning(f"❌ Session not found: {session_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    
    messages = chat_ops.get_messages_by_session(db, session_id)
    logger.info(f"✅ Retrieved {len(messages)} messages for session: {session_id}")
    return messages


@router.post("/message", response_model=chat_schemas.ChatResponse, status_code=status.HTTP_200_OK)
def chat(
    payload: chat_schemas.MessageCreate,
    db: Session = Depends(get_db),
    llm_service: LLMService = Depends(get_llm_service)
):
    """
    Send a message and get AI response with full LLM integration
    
    This endpoint:
    1. Validates the session exists
    2. Retrieves conversation history for context
    3. Saves user message to database
    4. Processes query through LLM service (intent detection + RAG)
    5. Saves AI response to database
    6. Returns structured response with metadata
    
    Args:
        payload: Message content and session ID
        db: Database session
        llm_service: LLM service instance
    
    Returns:
        Complete chat response with intent, answer, and metadata
    """
    logger.info(f"💬 New message in session {payload.session_id}: {payload.content[:100]}...")
    
    # 1. Verify session exists
    session = db.query(models.ChatSession).filter(
        models.ChatSession.id == payload.session_id
    ).first()
    
    if not session:
        logger.warning(f"❌ Session not found: {payload.session_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {payload.session_id} not found"
        )
    
    try:
        # 2. Get conversation history for context
        logger.debug(f"📚 Retrieving conversation history for context")
        conversation_history = chat_ops.get_conversation_history(
            db, 
            payload.session_id, 
            limit=10
        )
        
        # 3. Save user message to database FIRST
        user_msg = chat_ops.save_message(
            db=db,
            session_id=payload.session_id,
            role="human",
            content=payload.content
        )
        logger.info(f"💾 Saved user message: {user_msg.id}")
        
        # 4. Process query through LLM service (Intent + RAG + Answer)
        logger.info(f"🤖 Processing query with LLM service...")
        llm_result = llm_service.process_query(
            user_query=payload.content,
            conversation_history=conversation_history
        )
        
        # 5. Handle errors from LLM service
        if "error" in llm_result:
            error_msg = f"Sorry, I encountered an error processing your question: {llm_result['error']}"
            logger.error(f"❌ LLM processing error: {llm_result['error']}")
            ai_msg = chat_ops.save_message(
                db=db,
                session_id=payload.session_id,
                role="system",
                content=error_msg
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_msg
            )

        
        # 6. Extract intent and answer
        answer_data = llm_result.get("answer","I apologize, but I couldn't process your request.")
        raw_sources = llm_result.get("sources_used",[])
        
        logger.info(f"✅ LLM response generated, sources: {len(raw_sources)}")
        
        # Format sources properly for frontend
        sources = []
        if raw_sources and isinstance(raw_sources, list):
            for source in raw_sources:
                if isinstance(source, dict):
                    # Create a readable source string with all metadata
                    act = source.get('act', 'Unknown Act')
                    section = source.get('section', '')
                    chapter = source.get('chapter', '')
                    content = source.get('content', '')
                    score = source.get('score', 0)
                    
                    # Format: "Act Name | Chapter X | Section Y | Content snippet"
                    source_str = f"📜 {act}"
                    if chapter:
                        source_str += f" | Chapter: {chapter}"
                    if section:
                        source_str += f" | Section: {section}"
                    source_str += f" | Relevance: {score}"
                    if content:
                        source_str += f"\n📄 {content[:300]}{'...' if len(content) > 300 else ''}"
                    
                    sources.append(source_str)
                else:
                    # Fallback for string sources
                    sources.append(str(source))
        
        # 8. Save AI response to database
        ai_msg = chat_ops.save_message(
            db=db,
            session_id=payload.session_id,
            role="system",
            content=answer_data
        )
        logger.info(f"💾 Saved AI response: {ai_msg.id}")
        
        # 9. Build structured response
        response = chat_schemas.ChatResponse(
            session_id=payload.session_id,
            ai_response=answer_data,
            sources=sources
        )
        
        logger.info(f"✅ Successfully processed message in session {payload.session_id}")
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error processing message: {e}", exc_info=True)
        
        # Save error message to database
        error_response = "I apologize, but I encountered an unexpected error. Please try again."
        error_msg = chat_ops.save_message(
            db=db,
            session_id=payload.session_id,
            role="system",
            content=error_response
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.delete("/session/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(session_id: uuid.UUID, db: Session = Depends(get_db)):
    """
    Delete a chat session and all its messages
    
    Args:
        session_id: Session UUID to delete
        db: Database session
    """
    logger.info(f"🗑️ Delete session request: {session_id}")
    
    session = db.query(models.ChatSession).filter(
        models.ChatSession.id == session_id
    ).first()
    
    if not session:
        logger.warning(f"❌ Session not found for deletion: {session_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    
    try:
        db.delete(session)
        db.commit()
        logger.info(f"✅ Deleted session {session_id}")
        return {"message": "Session deleted successfully"}
    except Exception as e:
        logger.error(f"❌ Failed to delete session: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete session"
        )