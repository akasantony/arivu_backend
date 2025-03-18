"""
LLM service for the Arivu learning platform.

This module provides a service for interacting with various LLM providers
and generating responses for educational interactions.
"""

import os
from typing import Dict, List, Any, Optional, Union
import time
import json
import logging
from enum import Enum
import httpx
from pydantic import BaseModel, Field

from .memory import Memory
from .prompts import get_agent_prompt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    COHERE = "cohere"
    AZURE_OPENAI = "azure_openai"
    CUSTOM = "custom"

class LLMConfig(BaseModel):
    """Configuration for LLM service."""
    provider: LLMProvider = Field(default=LLMProvider.OPENAI)
    model_name: str = Field(default="gpt-3.5-turbo")
    api_key: Optional[str] = Field(default=None)
    api_base: Optional[str] = Field(default=None)
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None)
    timeout: int = Field(default=30)
    additional_params: Dict[str, Any] = Field(default_factory=dict)

class LLMService:
    """
    Service for generating responses from language models.
    
    Provides a unified interface to various LLM providers with methods
    specifically designed for educational interactions.
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize the LLM service.
        
        Args:
            config: LLM configuration (optional, will use environment variables if not provided)
        """
        self.config = config or self._load_config_from_env()
        self.client = httpx.Client(timeout=self.config.timeout)
        self._setup_provider()
    
    def _load_config_from_env(self) -> LLMConfig:
        """
        Load configuration from environment variables.
        
        Returns:
            LLMConfig instance
        """
        provider_str = os.getenv("LLM_PROVIDER", "openai").lower()
        provider = LLMProvider(provider_str) if provider_str in [p.value for p in LLMProvider] else LLMProvider.OPENAI
        
        return LLMConfig(
            provider=provider,
            model_name=os.getenv("LLM_MODEL_NAME", "gpt-3.5-turbo"),
            api_key=os.getenv(f"{provider_str.upper()}_API_KEY"),
            api_base=os.getenv(f"{provider_str.upper()}_API_BASE"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "0")) or None,
            timeout=int(os.getenv("LLM_TIMEOUT", "30"))
        )
    
    def _setup_provider(self) -> None:
        """Set up the LLM provider client."""
        if self.config.provider == LLMProvider.OPENAI:
            try:
                import openai
                openai.api_key = self.config.api_key
                if self.config.api_base:
                    openai.api_base = self.config.api_base
                self._provider_client = openai
            except ImportError:
                logger.error("OpenAI package not installed. Please install it with 'pip install openai'.")
                raise
        
        elif self.config.provider == LLMProvider.ANTHROPIC:
            try:
                import anthropic
                self._provider_client = anthropic.Anthropic(api_key=self.config.api_key)
            except ImportError:
                logger.error("Anthropic package not installed. Please install it with 'pip install anthropic'.")
                raise
        
        elif self.config.provider == LLMProvider.AZURE_OPENAI:
            try:
                import openai
                openai.api_type = "azure"
                openai.api_key = self.config.api_key
                openai.api_base = self.config.api_base or os.getenv("AZURE_OPENAI_ENDPOINT")
                openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
                self._provider_client = openai
            except ImportError:
                logger.error("OpenAI package not installed. Please install it with 'pip install openai'.")
                raise
        
        elif self.config.provider == LLMProvider.COHERE:
            try:
                import cohere
                self._provider_client = cohere.Client(api_key=self.config.api_key)
            except ImportError:
                logger.error("Cohere package not installed. Please install it with 'pip install cohere'.")
                raise
        
        elif self.config.provider == LLMProvider.HUGGINGFACE:
            # For HuggingFace, we'll use the httpx client directly
            pass
    
    def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate a chat completion using the configured provider.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            
        Returns:
            Response dictionary with 'message' and metadata
        """
        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens
        
        try:
            if self.config.provider == LLMProvider.OPENAI:
                return self._generate_openai_chat_completion(messages, temp, max_tok)
            
            elif self.config.provider == LLMProvider.ANTHROPIC:
                return self._generate_anthropic_chat_completion(messages, temp, max_tok)
            
            elif self.config.provider == LLMProvider.AZURE_OPENAI:
                return self._generate_azure_openai_chat_completion(messages, temp, max_tok)
                
            elif self.config.provider == LLMProvider.COHERE:
                return self._generate_cohere_chat_completion(messages, temp, max_tok)
                
            elif self.config.provider == LLMProvider.HUGGINGFACE:
                return self._generate_huggingface_chat_completion(messages, temp, max_tok)
            
            else:
                raise ValueError(f"Unsupported provider: {self.config.provider}")
                
        except Exception as e:
            logger.error(f"Error generating chat completion: {str(e)}")
            return {
                "message": f"I apologize, but I encountered an error: {str(e)}. Please try again later.",
                "error": str(e)
            }
    
    def _generate_openai_chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int]
    ) -> Dict[str, Any]:
        """
        Generate a chat completion using OpenAI.
        
        Args:
            messages: List of message dictionaries
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            Response dictionary
        """
        start_time = time.time()
        
        params = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": temperature
        }
        
        if max_tokens:
            params["max_tokens"] = max_tokens
            
        # Add any additional parameters from config
        params.update(self.config.additional_params)
        
        response = self._provider_client.ChatCompletion.create(**params)
        
        end_time = time.time()
        
        return {
            "message": response.choices[0].message.content,
            "provider": "openai",
            "model": self.config.model_name,
            "response_time": end_time - start_time,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
    
    def _generate_anthropic_chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int]
    ) -> Dict[str, Any]:
        """
        Generate a chat completion using Anthropic.
        
        Args:
            messages: List of message dictionaries
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            Response dictionary
        """
        start_time = time.time()
        
        # Convert messages to Anthropic format
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                # Anthropic doesn't have system messages, so we'll prepend to the first user message
                continue
            elif msg["role"] == "user":
                prompt += f"\n\nHuman: {msg['content']}"
            elif msg["role"] == "assistant":
                prompt += f"\n\nAssistant: {msg['content']}"
        
        # Add the final "Assistant: " to prompt the model to respond
        prompt += "\n\nAssistant: "
        
        # Find system message if it exists
        system_message = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
        
        params = {
            "prompt": prompt,
            "model": self.config.model_name,
            "temperature": temperature,
            "max_tokens_to_sample": max_tokens or 1000,
            "stop_sequences": ["\n\nHuman:"]
        }
        
        if system_message:
            params["system"] = system_message
            
        # Add any additional parameters from config
        params.update(self.config.additional_params)
        
        response = self._provider_client.completions.create(**params)
        
        end_time = time.time()
        
        return {
            "message": response.completion,
            "provider": "anthropic",
            "model": self.config.model_name,
            "response_time": end_time - start_time
        }
    
    def _generate_azure_openai_chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int]
    ) -> Dict[str, Any]:
        """
        Generate a chat completion using Azure OpenAI.
        
        Args:
            messages: List of message dictionaries
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            Response dictionary
        """
        start_time = time.time()
        
        # Azure OpenAI deployment name is stored in model_name
        deployment_name = self.config.model_name
        
        params = {
            "engine": deployment_name,
            "messages": messages,
            "temperature": temperature
        }
        
        if max_tokens:
            params["max_tokens"] = max_tokens
            
        # Add any additional parameters from config
        params.update(self.config.additional_params)
        
        response = self._provider_client.ChatCompletion.create(**params)
        
        end_time = time.time()
        
        return {
            "message": response.choices[0].message.content,
            "provider": "azure_openai",
            "model": deployment_name,
            "response_time": end_time - start_time,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
    
    def _generate_cohere_chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int]
    ) -> Dict[str, Any]:
        """
        Generate a chat completion using Cohere.
        
        Args:
            messages: List of message dictionaries
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            Response dictionary
        """
        start_time = time.time()
        
        # Convert messages to Cohere format
        chat_history = []
        system_message = None
        
        # Extract conversation history and system message
        for msg in messages[:-1]:  # Exclude the last message
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                chat_history.append({
                    "role": msg["role"],
                    "message": msg["content"]
                })
        
        # Get the current message (last one)
        current_message = messages[-1]["content"] if messages[-1]["role"] == "user" else None
        
        if not current_message:
            raise ValueError("The last message must be from the user")
        
        params = {
            "model": self.config.model_name,
            "message": current_message,
            "chat_history": chat_history,
            "temperature": temperature
        }
        
        if system_message:
            params["preamble"] = system_message
            
        if max_tokens:
            params["max_tokens"] = max_tokens
            
        # Add any additional parameters from config
        params.update(self.config.additional_params)
        
        response = self._provider_client.chat(**params)
        
        end_time = time.time()
        
        return {
            "message": response.text,
            "provider": "cohere",
            "model": self.config.model_name,
            "response_time": end_time - start_time,
            "token_count": getattr(response, "token_count", None)
        }
    
    def _generate_huggingface_chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int]
    ) -> Dict[str, Any]:
        """
        Generate a chat completion using HuggingFace Inference API.
        
        Args:
            messages: List of message dictionaries
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            Response dictionary
        """
        start_time = time.time()
        
        api_url = f"{self.config.api_base or 'https://api-inference.huggingface.co'}/models/{self.config.model_name}"
        
        # Convert messages to the format expected by the model
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        payload = {
            "inputs": {
                "messages": formatted_messages
            },
            "parameters": {
                "temperature": temperature,
                "return_full_text": False
            }
        }
        
        if max_tokens:
            payload["parameters"]["max_new_tokens"] = max_tokens
            
        # Add any additional parameters from config
        payload["parameters"].update(self.config.additional_params)
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        response = self.client.post(
            api_url,
            json=payload,
            headers=headers
        )
        
        response.raise_for_status()
        result = response.json()
        
        end_time = time.time()
        
        # The response format can vary by model on HuggingFace
        message = ""
        if isinstance(result, list) and len(result) > 0:
            if "generated_text" in result[0]:
                message = result[0]["generated_text"]
            elif "content" in result[0]:
                message = result[0]["content"]
        elif isinstance(result, dict):
            if "generated_text" in result:
                message = result["generated_text"]
            elif "content" in result:
                message = result["content"]
        
        return {
            "message": message,
            "provider": "huggingface",
            "model": self.config.model_name,
            "response_time": end_time - start_time,
            "raw_response": result
        }
    
    def generate_response(
        self,
        user_input: str,
        system_prompt: str,
        memory: Optional[Memory] = None,
        available_tools: List[Any] = None
    ) -> Dict[str, Any]:
        """
        Generate a response for an educational interaction.
        
        Args:
            user_input: User's message
            system_prompt: System prompt for the agent
            memory: Optional memory instance with conversation history
            available_tools: Optional list of tools the agent can use
            
        Returns:
            Response dictionary with message and metadata
        """
        messages = []
        
        # Add system prompt
        messages.append({
            "role": "system",
            "content": system_prompt
        })
        
        # Add conversation history if provided
        if memory:
            history = memory.get_chat_history()
            messages.extend(history)
        
        # Add current user input
        messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Generate response
        response = self.generate_chat_completion(messages)
        
        # If tools are available, check if we need to use any
        if available_tools and response.get("message"):
            response = self._process_tools(response, user_input, available_tools)
        
        return response
    
    def _process_tools(
        self,
        initial_response: Dict[str, Any],
        user_input: str,
        available_tools: List[Any]
    ) -> Dict[str, Any]:
        """
        Process tool usage in response.
        
        Checks if the response indicates a need to use tools,
        and if so, executes the appropriate tools and updates the response.
        
        Args:
            initial_response: Initial response from LLM
            user_input: User's message
            available_tools: List of available tools
            
        Returns:
            Updated response with tool results
        """
        message = initial_response["message"]
        tools_used = []
        sources = []
        
        # Simple heuristic to detect tool usage intent
        # In a real implementation, this would be more sophisticated
        for tool in available_tools:
            tool_name = getattr(tool, "name", str(tool))
            
            # Check if the response mentions using the tool
            if f"use {tool_name}" in message.lower() or f"using {tool_name}" in message.lower():
                try:
                    # Execute the tool
                    result = tool.run(user_input)
                    
                    # Track the tool usage
                    tools_used.append(tool_name)
                    
                    # If the tool provides sources, add them
                    if hasattr(result, "sources"):
                        sources.extend(result.sources)
                    
                    # Update the response with tool results
                    tool_result_msg = f"\n\nI used the {tool_name} to help with your question. "
                    tool_result_msg += f"Here's what I found: {result}"
                    
                    message += tool_result_msg
                    
                except Exception as e:
                    # Log tool execution error
                    logger.error(f"Error executing tool {tool_name}: {str(e)}")
                    
                    # Inform the user
                    message += f"\n\nI tried to use the {tool_name}, but encountered an error: {str(e)}"
        
        # Update the response
        initial_response["message"] = message
        initial_response["tools_used"] = tools_used if tools_used else None
        initial_response["sources"] = sources if sources else None
        
        return initial_response
    
    def generate_educational_response(
        self,
        user_input: str,
        agent_type: str,
        agent_name: str,
        subject_area: str,
        learning_level: Dict[str, Any],
        memory: Optional[Memory] = None,
        available_tools: List[Any] = None,
        additional_instructions: str = ""
    ) -> Dict[str, Any]:
        """
        Generate an educational response using a subject-specific agent.
        
        Args:
            user_input: User's message
            agent_type: Type of agent (math, science, language, history)
            agent_name: Name of the agent
            subject_area: Subject area description
            learning_level: Dictionary with learning level parameters
            memory: Optional memory instance with conversation history
            available_tools: Optional list of tools the agent can use
            additional_instructions: Additional custom instructions
            
        Returns:
            Response dictionary with message and metadata
        """
        # Generate the appropriate prompt for this agent
        system_prompt = get_agent_prompt(
            agent_type=agent_type,
            agent_name=agent_name,
            subject_area=subject_area,
            learning_level=learning_level,
            additional_instructions=additional_instructions
        )
        
        # Generate the response
        return self.generate_response(
            user_input=user_input,
            system_prompt=system_prompt,
            memory=memory,
            available_tools=available_tools
        )
    
    def analyze_learning_level(
        self,
        conversation_history: List[Dict[str, str]],
        current_level: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze conversation history to recommend appropriate learning level adjustments.
        
        Args:
            conversation_history: List of conversation messages
            current_level: Dictionary with current learning level parameters
            
        Returns:
            Dictionary with analysis and recommendations
        """
        # Construct a prompt for analyzing learning level
        prompt = f"""
        You are an educational assessment AI. Analyze the following conversation between a student and an educational agent.
        The student is currently at learning level: {current_level['name']} (complexity: {current_level['complexity']}/10).
        
        Based on the conversation, determine:
        1. If the current level is appropriate
        2. If the student seems to be struggling (needs a lower level)
        3. If the student seems to be advancing quickly (needs a higher level)
        4. Specific strengths or weaknesses demonstrated
        
        Conversation:
        """
        
        # Add the conversation history
        for msg in conversation_history:
            role = msg["role"]
            if role == "user":
                prompt += f"\nStudent: {msg['content']}\n"
            elif role == "assistant":
                prompt += f"\nEducational Agent: {msg['content']}\n"
        
        prompt += "\nAnalysis:"
        
        # Generate analysis
        messages = [
            {"role": "system", "content": "You are an educational assessment expert."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.generate_chat_completion(messages)
        
        # Parse the analysis to extract recommendations
        analysis = response.get("message", "")
        
        # Determine level adjustment (simple heuristic for demonstration)
        adjustment = 0
        if "struggling" in analysis.lower() or "simpler" in analysis.lower():
            adjustment = -1
        elif "advancing" in analysis.lower() or "higher level" in analysis.lower():
            adjustment = 1
        
        return {
            "analysis": analysis,
            "recommended_adjustment": adjustment,
            "current_level": current_level["name"],
            "current_complexity": current_level["complexity"]
        }
    
    def generate_learning_exercises(
        self,
        subject: str,
        topic: str,
        difficulty: int,
        count: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate learning exercises for a specific subject and topic.
        
        Args:
            subject: Subject area (math, science, language, history)
            topic: Specific topic within the subject
            difficulty: Difficulty level (1-10)
            count: Number of exercises to generate
            
        Returns:
            List of exercise dictionaries
        """
        prompt = f"""
        Create {count} educational exercises for a student learning about {topic} in {subject}.
        The difficulty level should be {difficulty} out of 10.
        
        For each exercise, provide:
        1. A clear problem statement or question
        2. The solution with explanation
        3. One or two hints that could help the student
        
        Format your response as a JSON array with objects containing:
        - "problem": The problem statement
        - "solution": The full solution with explanation
        - "hints": An array of hints
        - "difficulty": The actual difficulty level (1-10)
        """
        
        messages = [
            {"role": "system", "content": "You are an educational exercise generator."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.generate_chat_completion(messages)
        
        # Parse the JSON output
        try:
            # Extract JSON from the response (handling potential text before or after)
            message = response.get("message", "")
            json_start = message.find("[")
            json_end = message.rfind("]") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = message[json_start:json_end]
                exercises = json.loads(json_str)
                
                # Validate the structure
                valid_exercises = []
                for ex in exercises:
                    if "problem" in ex and "solution" in ex:
                        if "hints" not in ex:
                            ex["hints"] = []
                        if "difficulty" not in ex:
                            ex["difficulty"] = difficulty
                        valid_exercises.append(ex)
                
                return valid_exercises
            
            # Fallback: try to parse the entire message
            return json.loads(message)
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse exercises: {str(e)}")
            
            # Fallback to a simple parsing approach
            message = response.get("message", "")
            exercises = []
            
            # Simple parsing logic for when JSON parsing fails
            sections = message.split("Exercise")
            for i, section in enumerate(sections[1:], 1):
                if "Problem:" in section and "Solution:" in section:
                    problem_parts = section.split("Solution:")
                    problem = problem_parts[0].replace("Problem:", "").strip()
                    solution = problem_parts[1].strip()
                    
                    # Extract hints if present
                    hints = []
                    if "Hint:" in solution:
                        hint_parts = solution.split("Hint:")
                        solution = hint_parts[0].strip()
                        hints = [h.strip() for h in hint_parts[1:]]
                    
                    exercises.append({
                        "problem": problem,
                        "solution": solution,
                        "hints": hints,
                        "difficulty": difficulty
                    })
            
            return exercises[:count]
    
    def generate_personalized_learning_plan(
        self,
        subject: str,
        strengths: Dict[str, float],
        weaknesses: Dict[str, float],
        learning_pace: int,
        target_level: str
    ) -> Dict[str, Any]:
        """
        Generate a personalized learning plan based on student's performance.
        
        Args:
            subject: Subject area (math, science, language, history)
            strengths: Dictionary of strengths with scores
            weaknesses: Dictionary of weaknesses with scores
            learning_pace: Learning pace score (1-10)
            target_level: Target learning level
            
        Returns:
            Dictionary with personalized learning plan
        """
        # Format strengths and weaknesses for the prompt
        strengths_str = "\n".join([f"- {topic}: {score}" for topic, score in strengths.items()])
        weaknesses_str = "\n".join([f"- {topic}: {score}" for topic, score in weaknesses.items()])
        
        prompt = f"""
        Create a personalized learning plan for a student studying {subject}.
        
        Student Profile:
        - Learning Pace: {learning_pace}/10
        - Target Level: {target_level}
        
        Strengths:
        {strengths_str if strengths else "- None identified yet"}
        
        Areas for Improvement:
        {weaknesses_str if weaknesses else "- None identified yet"}
        
        The learning plan should include:
        1. Short-term goals (next 2 weeks)
        2. Medium-term goals (next 1-2 months)
        3. Recommended topics to focus on
        4. Specific learning activities
        5. Suggested resources
        6. A study schedule based on the student's learning pace
        
        Format the response as a structured learning plan that can be presented to the student.
        """
        
        messages = [
            {"role": "system", "content": "You are an educational planning expert."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.generate_chat_completion(messages)
        
        # Process the learning plan
        learning_plan = {
            "plan": response.get("message", ""),
            "subject": subject,
            "target_level": target_level,
            "created_at": time.time(),
            "strengths": strengths,
            "weaknesses": weaknesses,
            "learning_pace": learning_pace
        }
        
        return learning_plan
    
    def generate_suggested_questions(
        self,
        subject: str,
        recent_topics: List[str],
        conversation_history: List[Dict[str, str]],
        count: int = 3
    ) -> List[str]:
        """
        Generate suggested follow-up questions for the student.
        
        Args:
            subject: Subject area
            recent_topics: Recently discussed topics
            conversation_history: Recent conversation history
            count: Number of questions to generate
            
        Returns:
            List of suggested questions
        """
        # Format recent topics and conversation
        topics_str = ", ".join(recent_topics)
        
        conversation_str = ""
        for msg in conversation_history[-6:]:  # Use last 6 messages
            role = "Student" if msg["role"] == "user" else "Tutor"
            conversation_str += f"{role}: {msg['content']}\n\n"
        
        prompt = f"""
        Based on the following conversation about {subject}, generate {count} thoughtful follow-up questions 
        that the student might want to ask next to deepen their understanding.
        
        Recent topics: {topics_str}
        
        Recent conversation:
        {conversation_str}
        
        Generate {count} follow-up questions that:
        1. Build on what has been discussed
        2. Help clarify any potential points of confusion
        3. Explore interesting related concepts
        4. Are specific and clearly formulated
        5. Encourage deeper thinking
        
        Return ONLY the questions, one per line, without numbering or explanation.
        """
        
        messages = [
            {"role": "system", "content": "You generate helpful follow-up questions for educational conversations."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.generate_chat_completion(messages)
        
        # Process the response to extract questions
        questions_text = response.get("message", "").strip()
        questions = [q.strip() for q in questions_text.split("\n") if q.strip() and "?" in q]
        
        # Limit to requested count
        return questions[:count]
    
    def assess_response(
        self,
        subject: str,
        question: str,
        student_response: str,
        correct_answer: Optional[str] = None,
        learning_level: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Assess a student's response to a question.
        
        Args:
            subject: Subject area
            question: The question posed
            student_response: Student's response to assess
            correct_answer: Optional correct answer for comparison
            learning_level: Optional learning level parameters
            
        Returns:
            Assessment dictionary with feedback
        """
        level_info = ""
        if learning_level:
            level_name = learning_level.get("name", "Intermediate")
            complexity = learning_level.get("complexity", 5)
            level_info = f"The student is at {level_name} level (complexity {complexity}/10)."
        
        prompt = f"""
        Assess the following student response in {subject}:
        
        Question: {question}
        
        Student's Response: {student_response}
        """
        
        if correct_answer:
            prompt += f"\n\nCorrect Answer: {correct_answer}"
            
        prompt += f"""
        
        {level_info}
        
        Provide:
        1. An overall assessment of correctness (percentage or score out of 10)
        2. Specific feedback on what was correct
        3. Specific feedback on any misconceptions or errors
        4. Suggestions for improvement
        5. A short encouraging note
        
        Format your assessment in a clear, structured way that would be helpful for the student.
        """
        
        messages = [
            {"role": "system", "content": "You are an educational assessment expert who provides constructive feedback."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.generate_chat_completion(messages)
        
        # Extract a numerical score if present in the response
        assessment_text = response.get("message", "")
        score = None
        
        # Look for percentage or score patterns
        import re
        percentage_match = re.search(r'(\d{1,3})%', assessment_text)
        score_match = re.search(r'(\d{1,2})(\s)*/(\s)*10', assessment_text)
        
        if percentage_match:
            try:
                score = int(percentage_match.group(1)) / 100
            except (ValueError, IndexError):
                pass
        elif score_match:
            try:
                score_text = score_match.group(0).replace(" ", "")
                parts = score_text.split("/")
                if len(parts) == 2:
                    score = float(parts[0]) / float(parts[1])
            except (ValueError, IndexError):
                pass
        
        return {
            "assessment": assessment_text,
            "score": score,
            "subject": subject,
            "question": question,
            "student_response": student_response
        }
    
    def generate_concept_map(
        self,
        subject: str,
        central_concept: str,
        complexity: int = 5,
        max_concepts: int = 10
    ) -> Dict[str, Any]:
        """
        Generate a concept map for visualizing relationships between concepts.
        
        Args:
            subject: Subject area
            central_concept: The main concept for the map
            complexity: Complexity level (1-10)
            max_concepts: Maximum number of related concepts to include
            
        Returns:
            Dictionary with concept map data
        """
        prompt = f"""
        Create a concept map for teaching about "{central_concept}" in {subject}.
        The map should have complexity level {complexity}/10 and include up to {max_concepts} concepts.
        
        The concept map should show:
        1. The central concept: "{central_concept}"
        2. Related concepts that are important to understand
        3. Clear relationships between concepts (e.g., "leads to", "is part of", "influences")
        
        Format your response as a JSON object with:
        - "central_concept": The main concept
        - "nodes": Array of concept nodes, each with "id", "name", and "description"
        - "links": Array of relationships, each with "source" (node id), "target" (node id), and "relationship" (description)
        
        Make sure each concept is meaningful and connects to at least one other concept.
        """
        
        messages = [
            {"role": "system", "content": "You are an expert in creating educational concept maps."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.generate_chat_completion(messages)
        
        # Parse the JSON output
        try:
            # Extract JSON from the response
            message = response.get("message", "")
            json_start = message.find("{")
            json_end = message.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = message[json_start:json_end]
                concept_map = json.loads(json_str)
                
                return {
                    "concept_map": concept_map,
                    "subject": subject,
                    "complexity": complexity,
                    "format": "json"
                }
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse concept map JSON: {str(e)}")
        
        # Fallback: return the raw text if JSON parsing fails
        return {
            "concept_map_text": response.get("message", ""),
            "subject": subject,
            "complexity": complexity,
            "format": "text"
        }
    
    def detect_knowledge_gaps(
        self,
        subject: str,
        conversation_history: List[Dict[str, str]],
        expected_knowledge: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze conversation to detect knowledge gaps.
        
        Args:
            subject: Subject area
            conversation_history: Conversation history
            expected_knowledge: List of concepts the student should understand
            
        Returns:
            Dictionary with detected knowledge gaps and recommendations
        """
        # Format conversation history
        conversation_str = ""
        for msg in conversation_history:
            role = "Student" if msg["role"] == "user" else "Tutor"
            conversation_str += f"{role}: {msg['content']}\n\n"
        
        # Format expected knowledge
        expected_knowledge_str = "\n".join([f"- {concept}" for concept in expected_knowledge])
        
        prompt = f"""
        You are an educational diagnostician. Analyze the following conversation in {subject} 
        to identify potential knowledge gaps compared to expected knowledge.
        
        Expected Knowledge:
        {expected_knowledge_str}
        
        Conversation:
        {conversation_str}
        
        For each expected knowledge item, assess:
        1. Whether the student demonstrates understanding
        2. Whether there are misconceptions or gaps
        3. The confidence of your assessment (high, medium, low)
        
        Then provide:
        1. A list of probable knowledge gaps
        2. Recommended topics to review
        3. Suggested questions to confirm these gaps
        
        Format your response as a JSON object with "assessments", "gaps", "review_topics", and "confirmation_questions".
        """
        
        messages = [
            {"role": "system", "content": "You are an expert educational diagnostician."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.generate_chat_completion(messages)
        
        # Parse the JSON output
        try:
            # Extract JSON from the response
            message = response.get("message", "")
            json_start = message.find("{")
            json_end = message.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = message[json_start:json_end]
                analysis = json.loads(json_str)
                
                return {
                    "analysis": analysis,
                    "subject": subject,
                    "raw_response": False
                }
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse knowledge gap analysis JSON: {str(e)}")
        
        # Fallback: return the raw text if JSON parsing fails
        return {
            "analysis_text": response.get("message", ""),
            "subject": subject,
            "raw_response": True
        }
    
    def close(self):
        """Close the HTTP client."""
        self.client.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()