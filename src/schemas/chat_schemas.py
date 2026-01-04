from pydantic import BaseModel
from uuid import UUID
from datetime import datetime
from typing import List, Optional, Dict, Any

class MessageCreate(BaseModel):
    session_id: UUID
    content: str

class MessageResponse(BaseModel):
    id: UUID
    role: str
    content: str
    created_at: datetime
    class Config:
        from_attributes = True

class IntentDetail(BaseModel):
    """Intent detection details"""
    language: str
    is_follow_up: bool
    enhanced_question: Optional[str] = None
    intent: str
    cleaned_question: str

class AnswerDetail(BaseModel):
    """Legal answer details"""
    answer: str
    confidence: str
    sources_used: List[str] = []
    disclaimer: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: UUID
    ai_response: str
    sources: List[str] = []  
    
class SessionInfo(BaseModel):
    """Session information"""
    session_id: UUID
    user_id: str
    created_at: datetime
    message_count: int