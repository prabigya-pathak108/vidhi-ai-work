"""
Test Prompt Factory - validates prompt loading
"""
import pytest
from src.core.prompt_factory import PromptFactory


def test_prompt_factory_loads_intent_prompt():
    """Test that intent detection prompt can be loaded with required variables"""
    prompt = PromptFactory.get_prompt(
        "identify_intent",
        conversation_history="",
        user_question="Test question"
    )
    
    assert prompt is not None
    assert len(prompt) > 0
    assert isinstance(prompt, str)


def test_prompt_factory_loads_legal_answer_prompt():
    """Test that legal answer prompt can be loaded with required variables"""
    prompt = PromptFactory.get_prompt(
        "answer_legal_question",
        conversation_history="",
        question="Test question",
        intented_question_result="Test intent result",
        context="Test context",
        format_instructions="JSON format"
    )
    
    assert prompt is not None
    assert len(prompt) > 0
    assert isinstance(prompt, str)


def test_prompt_factory_raises_error_for_invalid_prompt():
    """Test that invalid prompt name raises error"""
    with pytest.raises(FileNotFoundError):
        PromptFactory.get_prompt("non_existent_prompt")
