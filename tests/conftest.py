"""
Simple test configuration for Vidhi AI
"""
import pytest
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Set test environment variables
os.environ['DATABASE_URL'] = 'postgresql://postgres:password@localhost:5432/vidhi_ai_test'
os.environ['PINECONE_API_KEY'] = 'test-key'
os.environ['GROQ_API_KEY'] = 'test-key'
os.environ['GEMINI_API_KEY'] = 'test-key'
