"""
Memory management for conversation history in the Arivu platform.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json

class Memory:
    """
    Class for managing conversation history between users and agents.
    
    Stores interactions and provides methods to retrieve conversation
    history in various formats for LLM context.
    """
    
    def __init__(self, max_history: int = 10):
        """
        Initialize memory storage.
        
        Args:
            max_history: Maximum number of interactions to store
        """
        self.interactions: List[Dict[str, Any]] = []
        self.max_history = max_history
    
    def add_interaction(self, user_message: Optional[str], assistant_message: Optional[str]) -> None:
        """
        Add an interaction to memory.
        
        Args:
            user_message: User's message (can be None if only adding assistant message)
            assistant_message: Assistant's response (can be None if only adding user message)
        """
        if not user_message and not assistant_message:
            return
            
        timestamp = datetime.utcnow().isoformat()
        
        interaction = {
            "timestamp": timestamp
        }
        
        if user_message:
            interaction["user"] = user_message
            
        if assistant_message:
            interaction["assistant"] = assistant_message
        
        self.interactions.append(interaction)
        
        # Trim history if needed
        if len(self.interactions) > self.max_history:
            self.interactions = self.interactions[-self.max_history:]
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """
        Get conversation history in a format suitable for chat-based LLMs.
        
        Returns:
            List of message dictionaries with 'role' and 'content'
        """
        history = []
        
        for interaction in self.interactions:
            if "user" in interaction:
                history.append({
                    "role": "user",
                    "content": interaction["user"]
                })
            
            if "assistant" in interaction:
                history.append({
                    "role": "assistant",
                    "content": interaction["assistant"]
                })
        
        return history
    
    def get_recent_messages(self, count: int = 5) -> List[Dict[str, str]]:
        """
        Get the most recent messages in chat format.
        
        Args:
            count: Number of recent messages to retrieve
            
        Returns:
            List of recent message dictionaries
        """
        history = self.get_chat_history()
        return history[-count:] if len(history) > count else history
    
    def get_string_history(self, include_roles: bool = True) -> str:
        """
        Get conversation history as a formatted string.
        
        Args:
            include_roles: Whether to include role labels in the output
            
        Returns:
            Formatted conversation history string
        """
        history_str = ""
        
        for interaction in self.interactions:
            if "user" in interaction:
                if include_roles:
                    history_str += f"User: {interaction['user']}\n\n"
                else:
                    history_str += f"{interaction['user']}\n\n"
            
            if "assistant" in interaction:
                if include_roles:
                    history_str += f"Assistant: {interaction['assistant']}\n\n"
                else:
                    history_str += f"{interaction['assistant']}\n\n"
        
        return history_str.strip()
    
    def clear(self) -> None:
        """Clear all stored interactions."""
        self.interactions = []
    
    def save_to_json(self) -> str:
        """
        Save memory to a JSON string.
        
        Returns:
            JSON string representation of memory
        """
        return json.dumps({
            "max_history": self.max_history,
            "interactions": self.interactions
        })
    
    @classmethod
    def load_from_json(cls, json_str: str) -> 'Memory':
        """
        Load memory from a JSON string.
        
        Args:
            json_str: JSON string representation of memory
            
        Returns:
            Memory instance with loaded data
        """
        data = json.loads(json_str)
        memory = cls(max_history=data.get("max_history", 10))
        memory.interactions = data.get("interactions", [])
        return memory
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the conversation.
        
        Returns:
            Dictionary with conversation metrics
        """
        if not self.interactions:
            return {
                "message_count": 0,
                "user_messages": 0,
                "assistant_messages": 0,
                "duration": 0
            }
            
        first_timestamp = datetime.fromisoformat(self.interactions[0]["timestamp"])
        last_timestamp = datetime.fromisoformat(self.interactions[-1]["timestamp"])
        duration = (last_timestamp - first_timestamp).total_seconds()
        
        user_messages = sum(1 for interaction in self.interactions if "user" in interaction)
        assistant_messages = sum(1 for interaction in self.interactions if "assistant" in interaction)
        
        return {
            "message_count": user_messages + assistant_messages,
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "duration_seconds": duration,
            "first_message": first_timestamp.isoformat(),
            "last_message": last_timestamp.isoformat()
        }