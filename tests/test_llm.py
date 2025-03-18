"""
Tests for the LLM module in the Arivu learning platform.
"""

import pytest
from unittest.mock import patch, MagicMock
import json
import os
from typing import Dict, Any, List

from arivu.llm.service import LLMService, LLMConfig, LLMProvider
from arivu.llm.memory import Memory
from arivu.llm.prompts import get_prompt_template, register_prompt_template, format_prompt

class TestMemory:
    """Tests for the Memory class."""
    
    def test_add_interaction(self):
        """Test adding interactions to memory."""
        memory = Memory()
        memory.add_interaction("Hello", "Hi there")
        memory.add_interaction("How are you?", "I'm doing well, thanks!")
        
        assert len(memory.interactions) == 2
        assert memory.interactions[0]["user"] == "Hello"
        assert memory.interactions[0]["assistant"] == "Hi there"
        assert memory.interactions[1]["user"] == "How are you?"
        assert memory.interactions[1]["assistant"] == "I'm doing well, thanks!"
    
    def test_get_chat_history(self):
        """Test getting chat history in message format."""
        memory = Memory()
        memory.add_interaction("Hello", "Hi there")
        
        history = memory.get_chat_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "Hi there"
    
    def test_get_string_history(self):
        """Test getting history as a formatted string."""
        memory = Memory()
        memory.add_interaction("Hello", "Hi there")
        
        history_with_roles = memory.get_string_history(include_roles=True)
        assert "User: Hello" in history_with_roles
        assert "Assistant: Hi there" in history_with_roles
        
        history_without_roles = memory.get_string_history(include_roles=False)
        assert "User:" not in history_without_roles
        assert "Assistant:" not in history_without_roles
        assert "Hello" in history_without_roles
        assert "Hi there" in history_without_roles
    
    def test_clear_memory(self):
        """Test clearing memory."""
        memory = Memory()
        memory.add_interaction("Hello", "Hi there")
        assert len(memory.interactions) == 1
        
        memory.clear()
        assert len(memory.interactions) == 0
    
    def test_save_load_json(self):
        """Test saving and loading memory to/from JSON."""
        memory = Memory()
        memory.add_interaction("Hello", "Hi there")
        
        json_str = memory.save_to_json()
        loaded_memory = Memory.load_from_json(json_str)
        
        assert len(loaded_memory.interactions) == 1
        assert loaded_memory.interactions[0]["user"] == "Hello"
        assert loaded_memory.interactions[0]["assistant"] == "Hi there"
    
    def test_memory_limit(self):
        """Test memory respects max_history limit."""
        memory = Memory(max_history=2)
        
        memory.add_interaction("Message 1", "Response 1")
        memory.add_interaction("Message 2", "Response 2")
        memory.add_interaction("Message 3", "Response 3")
        
        assert len(memory.interactions) == 2
        assert memory.interactions[0]["user"] == "Message 2"
        assert memory.interactions[1]["user"] == "Message 3"

class TestPrompts:
    """Tests for prompt management."""
    
    def test_get_existing_prompt(self):
        """Test getting an existing prompt template."""
        math_prompt = get_prompt_template("math_agent")
        assert math_prompt is not None
        assert "math education" in math_prompt.lower()
    
    def test_register_custom_prompt(self):
        """Test registering a custom prompt template."""
        custom_prompt = "This is a custom prompt for ${subject} with ${variable}."
        register_prompt_template("test_custom", custom_prompt)
        
        retrieved_prompt = get_prompt_template("test_custom")
        assert retrieved_prompt == custom_prompt
    
    def test_format_prompt(self):
        """Test formatting a prompt template with variables."""
        custom_prompt = "This is a custom prompt for ${subject} with ${variable}."
        register_prompt_template("test_format", custom_prompt)
        
        formatted = format_prompt("test_format", subject="science", variable="substitution")
        assert formatted == "This is a custom prompt for science with substitution."
    
    def test_agent_prompt_generation(self):
        """Test generating a complete agent prompt."""
        from arivu.llm.prompts import get_agent_prompt
        
        agent_prompt = get_agent_prompt(
            agent_type="math",
            agent_name="Test Agent",
            subject_area="test mathematics",
            learning_level={
                "name": "Beginner",
                "complexity": 3,
                "vocabulary_level": 3,
                "depth_of_explanation": 4,
                "example_frequency": 8,
                "assumed_prior_knowledge": 2
            }
        )
        
        assert "Test Agent" in agent_prompt
        assert "test mathematics" in agent_prompt
        assert "Beginner" in agent_prompt
        assert "complexity: 3" in agent_prompt.lower()

@pytest.fixture
def mock_llm_service():
    """Fixture for a mocked LLM service."""
    with patch('arivu.llm.service.LLMService._generate_openai_chat_completion') as mock_generate:
        # Mock the OpenAI completion method
        mock_generate.return_value = {
            "message": "This is a test response from the mock LLM service.",
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "response_time": 0.5,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
        
        # Create the service with the mocked method
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-3.5-turbo",
            api_key="fake-api-key"
        )
        service = LLMService(config=config)
        
        yield service

class TestLLMService:
    """Tests for the LLM service."""
    
    def test_generate_response(self, mock_llm_service):
        """Test generating a response with the LLM service."""
        response = mock_llm_service.generate_response(
            user_input="Test question",
            system_prompt="You are a test assistant.",
            memory=Memory()
        )
        
        assert "test response" in response["message"].lower()
        assert response["provider"] == "openai"
    
    def test_educational_response(self, mock_llm_service):
        """Test generating an educational response."""
        response = mock_llm_service.generate_educational_response(
            user_input="What is a quadratic equation?",
            agent_type="math",
            agent_name="Math Tutor",
            subject_area="mathematics",
            learning_level={
                "name": "Intermediate",
                "complexity": 5,
                "vocabulary_level": 5,
                "depth_of_explanation": 5,
                "example_frequency": 5,
                "assumed_prior_knowledge": 5
            }
        )
        
        assert "test response" in response["message"].lower()
    
    @patch('arivu.llm.service.LLMService.generate_chat_completion')
    def test_analyze_learning_level(self, mock_chat_completion, mock_llm_service):
        """Test analyzing learning level."""
        mock_chat_completion.return_value = {
            "message": "The student seems to be advancing quickly. I recommend increasing the complexity level."
        }
        
        conversation_history = [
            {"role": "user", "content": "What is calculus used for?"},
            {"role": "assistant", "content": "Calculus is used to study rates of change..."},
            {"role": "user", "content": "That makes sense! Can you explain derivatives?"}
        ]
        
        analysis = mock_llm_service.analyze_learning_level(
            conversation_history=conversation_history,
            current_level={"name": "Intermediate", "complexity": 5}
        )
        
        assert "advancing" in analysis["analysis"]
        assert analysis["recommended_adjustment"] == 1
    
    @patch('arivu.llm.service.LLMService.generate_chat_completion')
    def test_generate_learning_exercises(self, mock_chat_completion, mock_llm_service):
        """Test generating learning exercises."""
        mock_chat_completion.return_value = {
            "message": json.dumps([
                {
                    "problem": "Test problem 1",
                    "solution": "Test solution 1",
                    "hints": ["Hint 1", "Hint 2"],
                    "difficulty": 5
                },
                {
                    "problem": "Test problem 2",
                    "solution": "Test solution 2",
                    "hints": ["Hint 1"],
                    "difficulty": 6
                }
            ])
        }
        
        exercises = mock_llm_service.generate_learning_exercises(
            subject="math",
            topic="algebra",
            difficulty=5,
            count=2
        )
        
        assert len(exercises) == 2
        assert exercises[0]["problem"] == "Test problem 1"
        assert exercises[1]["solution"] == "Test solution 2"