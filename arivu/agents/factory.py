from typing import Dict, Any, Optional, Type

from arivu.agents.base_agent import BaseAgent
from arivu.agents.subject_agents.math_agent import MathAgent
from arivu.agents.subject_agents.science_agent import ScienceAgent
from arivu.agents.subject_agents.language_agent import LanguageAgent
from arivu.agents.subject_agents.history_agent import HistoryAgent

from arivu.llm.service import LLMService
from arivu.llm.memory import Memory
from arivu.tools.registry import ToolRegistry

class AgentFactory:
    """
    Factory class for creating different types of educational agents.
    """
    
    # Map agent types to agent classes
    AGENT_TYPES = {
        "math": MathAgent,
        "science": ScienceAgent,
        "language": LanguageAgent,
        "history": HistoryAgent
    }
    
    @classmethod
    def create_agent(
        cls,
        agent_type: str,
        agent_id: str,
        llm_service: LLMService,
        memory: Optional[Memory] = None,
        tool_registry: Optional[ToolRegistry] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> BaseAgent:
        """
        Create and return an agent of the specified type.
        
        Args:
            agent_type: Type of agent to create ('math', 'science', 'language', 'history')
            agent_id: Unique identifier for the agent
            llm_service: LLM service for the agent to use
            memory: Optional memory instance for the agent
            tool_registry: Optional tool registry for the agent
            config: Optional additional configuration parameters
            
        Returns:
            BaseAgent: An instance of the requested agent type
            
        Raises:
            ValueError: If the requested agent type is not supported
        """
        if agent_type not in cls.AGENT_TYPES:
            raise ValueError(f"Unsupported agent type: {agent_type}. "
                             f"Supported types are: {', '.join(cls.AGENT_TYPES.keys())}")
        
        # Create memory if not provided
        if memory is None:
            memory = Memory()
        
        # Create tool registry if not provided
        if tool_registry is None:
            tool_registry = ToolRegistry()
        
        # Get the agent class
        agent_class = cls.AGENT_TYPES[agent_type]
        
        # Create the agent with type-specific configurations
        if agent_type == "language" and config and "target_language" in config:
            return agent_class(
                agent_id=agent_id,
                llm_service=llm_service,
                memory=memory,
                tool_registry=tool_registry,
                target_language=config["target_language"]
            )
        else:
            return agent_class(
                agent_id=agent_id,
                llm_service=llm_service,
                memory=memory,
                tool_registry=tool_registry
            )
    
    @classmethod
    def register_agent_type(cls, agent_type: str, agent_class: Type[BaseAgent]) -> None:
        """
        Register a new agent type.
        
        Args:
            agent_type: The type identifier for the agent
            agent_class: The agent class to associate with this type
            
        Raises:
            ValueError: If the agent type is already registered
        """
        if agent_type in cls.AGENT_TYPES:
            raise ValueError(f"Agent type '{agent_type}' is already registered")
        
        cls.AGENT_TYPES[agent_type] = agent_class