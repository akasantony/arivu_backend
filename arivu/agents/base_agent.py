from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from abc import ABC, abstractmethod

from arivu.llm.service import LLMService
from arivu.llm.memory import Memory
from arivu.tools.registry import ToolRegistry

class AgentResponse(BaseModel):
    """Response model from an agent interaction."""
    message: str
    sources: Optional[List[Dict[str, Any]]] = None
    tools_used: Optional[List[str]] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class LearningLevel(BaseModel):
    """Model representing a learning level with specific parameters."""
    id: str
    name: str
    description: str
    complexity: int  # 1-10 scale
    vocabulary_level: int  # 1-10 scale
    depth_of_explanation: int  # 1-10 scale
    example_frequency: int  # 1-10 scale
    assumed_prior_knowledge: int  # 1-10 scale

class BaseAgent(ABC):
    """
    Base agent class that all subject agents inherit from.
    Provides common functionality for agents.
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str,
        llm_service: LLMService,
        memory: Memory,
        tool_registry: ToolRegistry,
        learning_levels: List[LearningLevel],
        default_level: str = "intermediate"
    ):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.llm_service = llm_service
        self.memory = memory
        self.tool_registry = tool_registry
        
        # Set up learning levels as a dictionary for quick access
        self.learning_levels = {level.id: level for level in learning_levels}
        
        # Set the default learning level
        if default_level not in self.learning_levels:
            default_level = list(self.learning_levels.keys())[0]
        self.current_level = self.learning_levels[default_level]
        
        # Keep track of user progress metrics
        self.user_metrics = {
            "questions_asked": 0,
            "average_response_time": 0,
            "topic_strengths": {},
            "topic_weaknesses": {},
            "learning_pace": 5,  # 1-10 scale
        }
        
        # Initialize the system prompt
        self._initialize_system_prompt()
    
    @abstractmethod
    def _get_subject_specific_instructions(self) -> str:
        """Get subject-specific instructions for the agent."""
        pass
    
    @abstractmethod
    def _get_subject_tools(self) -> List[str]:
        """Get a list of tool IDs relevant to this subject."""
        pass
    
    def _initialize_system_prompt(self) -> None:
        """Initialize the system prompt based on the current learning level."""
        base_prompt = f"""
        You are {self.name}, an educational AI assistant specializing in {self.description}.
        Your goal is to help students learn and understand concepts in an engaging way.
        
        Current learning level: {self.current_level.name}
        - Complexity: {self.current_level.complexity}/10
        - Vocabulary level: {self.current_level.vocabulary_level}/10
        - Depth of explanation: {self.current_level.depth_of_explanation}/10
        - Example frequency: {self.current_level.example_frequency}/10
        - Assumed prior knowledge: {self.current_level.assumed_prior_knowledge}/10
        
        {self._get_subject_specific_instructions()}
        
        Always adapt to the student's pace and understanding. If they're struggling, 
        provide simpler explanations. If they seem to grasp concepts quickly, 
        challenge them with more advanced material.
        
        Remember to:
        1. Use appropriate vocabulary for the current learning level
        2. Provide examples at the frequency appropriate for the current level
        3. Explain concepts with the appropriate depth
        4. Build on prior knowledge appropriate for the current level
        5. Encourage critical thinking and problem-solving
        6. Be encouraging and supportive
        """
        
        self.system_prompt = base_prompt
    
    def set_learning_level(self, level_id: str) -> bool:
        """
        Set the learning level for this agent.
        
        Args:
            level_id: The ID of the learning level to set
            
        Returns:
            bool: True if successful, False if the level was not found
        """
        if level_id in self.learning_levels:
            self.current_level = self.learning_levels[level_id]
            self._initialize_system_prompt()
            return True
        return False
    
    def generate_response(self, user_input: str) -> AgentResponse:
        """
        Generate a response to user input.
        
        Args:
            user_input: The user's message
            
        Returns:
            AgentResponse: The agent's response
        """
        # Update metrics
        self.user_metrics["questions_asked"] += 1
        
        # Get relevant tools for this interaction
        available_tools = [
            self.tool_registry.get_tool(tool_id) 
            for tool_id in self._get_subject_tools()
        ]
        
        # Generate response using LLM service
        llm_response = self.llm_service.generate_response(
            user_input=user_input,
            system_prompt=self.system_prompt,
            memory=self.memory,
            available_tools=available_tools
        )
        
        # Update memory with this interaction
        self.memory.add_interaction(user_input, llm_response["message"])
        
        # Adapt learning level based on interaction (simplified version)
        self._adapt_to_user_pace(user_input, llm_response)
        
        # Construct and return response
        return AgentResponse(
            message=llm_response["message"],
            sources=llm_response.get("sources"),
            tools_used=llm_response.get("tools_used"),
            confidence=llm_response.get("confidence"),
            metadata={
                "learning_level": self.current_level.id,
                "subject": self.description
            }
        )
    
    def _adapt_to_user_pace(self, user_input: str, llm_response: Dict[str, Any]) -> None:
        """
        Adapt the learning level based on user interactions.
        This is a simplified version that could be enhanced with more sophisticated analysis.
        
        Args:
            user_input: The user's message
            llm_response: The LLM's response data
        """
        # Example implementation - could be enhanced with ML-based analysis
        # Look for indicators of confusion
        confusion_indicators = [
            "i don't understand", "confused", "not clear", "what do you mean",
            "could you explain", "i'm lost", "that's too complex"
        ]
        
        # Look for indicators of advanced understanding
        advanced_indicators = [
            "that's simple", "i already know", "what about", "more advanced",
            "can we go deeper", "tell me more about", "interesting"
        ]
        
        # Simple scoring
        confusion_score = sum(1 for indicator in confusion_indicators if indicator in user_input.lower())
        advanced_score = sum(1 for indicator in advanced_indicators if indicator in user_input.lower())
        
        # Update learning pace
        pace_delta = advanced_score - confusion_score
        self.user_metrics["learning_pace"] = max(1, min(10, self.user_metrics["learning_pace"] + pace_delta * 0.5))
        
        # Automatically adjust level if significant change in pace
        # This is just a simple example, a real implementation might be more nuanced
        if self.user_metrics["learning_pace"] < 3 and self.current_level.complexity > 1:
            # Find a lower level
            for level in self.learning_levels.values():
                if level.complexity < self.current_level.complexity:
                    self.set_learning_level(level.id)
                    break
        elif self.user_metrics["learning_pace"] > 8 and self.current_level.complexity < 10:
            # Find a higher level
            for level in self.learning_levels.values():
                if level.complexity > self.current_level.complexity:
                    self.set_learning_level(level.id)
                    break

    def analyze_strengths_weaknesses(self) -> Dict[str, Any]:
        """
        Analyze the student's strengths and weaknesses based on interactions.
        
        Returns:
            Dict: Analysis of student's performance
        """
        # This would typically use more sophisticated analysis
        return {
            "strengths": self.user_metrics["topic_strengths"],
            "weaknesses": self.user_metrics["topic_weaknesses"],
            "learning_pace": self.user_metrics["learning_pace"],
            "recommended_focus_areas": list(self.user_metrics["topic_weaknesses"].keys())[:3]
        }