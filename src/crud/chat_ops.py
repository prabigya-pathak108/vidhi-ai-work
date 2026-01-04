from sqlalchemy.orm import Session
from src.database import models
import uuid
from typing import List
from src.utils.logger import get_logger

logger = get_logger(__name__)

def create_chat_session(db: Session, user_id: str):
    logger.info(f"Creating new chat session for user: {user_id}")
    new_session = models.ChatSession(user_id=user_id)
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    logger.info(f"✅ Chat session created: {new_session.id}")
    return new_session

def save_message(db: Session, session_id: uuid.UUID, role: str, content: str):
    logger.debug(f"Saving message to session {session_id}: role={role}, content_length={len(content)}")
    message = models.ChatMessage(session_id=session_id, role=role, content=content)
    db.add(message)
    db.commit()
    db.refresh(message)
    logger.debug(f"✅ Message saved: {message.id}")
    return message

def get_messages_by_session(db: Session, session_id: uuid.UUID):
    logger.debug(f"Retrieving messages for session: {session_id}")
    messages = db.query(models.ChatMessage).filter(
        models.ChatMessage.session_id == session_id
    ).order_by(models.ChatMessage.created_at.asc()).all()
    logger.debug(f"✅ Retrieved {len(messages)} messages")
    return messages

def get_conversation_history(db: Session, session_id: uuid.UUID, limit: int = 10) -> List[str]:
    """
    Get formatted conversation history for context
    
    Args:
        db: Database session
        session_id: Chat session ID
        limit: Maximum number of messages to retrieve
    
    Returns:
        List of formatted conversation strings
    """
    logger.debug(f"Getting conversation history for session: {session_id} (limit={limit})")
    messages = db.query(models.ChatMessage).filter(
        models.ChatMessage.session_id == session_id
    ).order_by(models.ChatMessage.created_at.desc()).limit(limit).all()
    
    # Reverse to get chronological order
    messages = list(reversed(messages))
    
    # Format as "Human: question" or "AI: answer"
    formatted = []
    for msg in messages:
        prefix = "Human" if msg.role == "human" else "AI"
        formatted.append(f"{prefix}: {msg.content}")
    
    logger.debug(f"✅ Retrieved {len(formatted)} conversation messages")
    return formatted

def get_session_with_count(db: Session, session_id: uuid.UUID):
    """Get session info with message count"""
    logger.debug(f"Getting session info with count: {session_id}")
    session = db.query(models.ChatSession).filter(
        models.ChatSession.id == session_id
    ).first()
    
    if not session:
        logger.warning(f"❌ Session not found: {session_id}")
        return None
    
    message_count = db.query(models.ChatMessage).filter(
        models.ChatMessage.session_id == session_id
    ).count()
    
    logger.debug(f"✅ Session found: {session_id}, message_count={message_count}")
    return {
        "session": session,
        "message_count": message_count
    }