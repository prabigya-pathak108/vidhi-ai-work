"""
Test Database Models - validates model structure
"""
import pytest
import uuid
from src.database.models import User, ChatSession, ChatMessage


def test_user_model_has_required_fields():
    """Test User model can be created with required fields"""
    user = User(id="test_user_123", name="Test User")
    
    assert user.id == "test_user_123"
    assert user.name == "Test User"


def test_chat_session_links_to_user():
    """Test ChatSession links to user_id"""
    session = ChatSession(user_id="test_user_123")
    
    assert session.user_id == "test_user_123"


def test_chat_message_has_content_and_role():
    """Test ChatMessage stores content and role"""
    session_id = uuid.uuid4()
    message = ChatMessage(
        session_id=session_id,
        role="human",
        content="What is property law?"
    )
    
    assert message.role == "human"
    assert message.content == "What is property law?"
    assert message.session_id == session_id
