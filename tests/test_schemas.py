"""
Test Pydantic Schemas - validates data validation
"""
import pytest
import uuid
from pydantic import ValidationError
from src.schemas.user_schemas import UserCreate, UserResponse
from src.schemas.chat_schemas import MessageCreate, MessageResponse


def test_user_create_validates_input():
    """Test UserCreate schema validates user data"""
    user = UserCreate(id="user123", name="Test User")
    
    assert user.id == "user123"
    assert user.name == "Test User"


def test_message_create_validates_content():
    """Test MessageCreate validates message data"""
    session_id = uuid.uuid4()
    
    message = MessageCreate(
        session_id=session_id,
        content="What is property law?"
    )
    
    assert message.content == "What is property law?"
    assert message.session_id == session_id


def test_message_create_accepts_uuid():
    """Test MessageCreate accepts valid UUID"""
    session_id = uuid.uuid4()
    
    message = MessageCreate(
        session_id=session_id,
        content="Test content"
    )
    
    assert message.session_id == session_id
    assert isinstance(message.session_id, uuid.UUID)
