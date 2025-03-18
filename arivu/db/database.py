import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Dict, Optional, Any, Union
from uuid import UUID

from arivu.llm.service import LLMService
from arivu.llm.memory import Memory
from arivu.tools.registry import ToolRegistry
from arivu.agents.factory import AgentFactory
from arivu.agents.base_agent import BaseAgent

# Create SQLAlchemy connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost/arivu")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Agent instance cache to avoid recreating them frequently
# Keys are agent_id strings, values are (agent_instance, last_used_timestamp) tuples
agent_cache: Dict[str, tuple] = {}

def get_db():
    """Get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_agent_instance(
    db: Session,
    agent_id: UUID,
    llm_service: LLMService,
    tool_registry: Optional[ToolRegistry] = None,
    reset_cache: bool = False
) -> BaseAgent:
    """
    Get or create an agent instance for a specific agent ID.
    
    Args:
        db: Database session
        agent_id: Agent ID
        llm_service: LLM service to use
        tool_registry: Optional tool registry to use
        reset_cache: Whether to force a fresh agent instance
        
    Returns:
        Agent instance
    """
    from arivu.db.models import Agent
    
    # Convert UUID to string for cache key
    cache_key = str(agent_id)
    
    # Check cache if not resetting
    if not reset_cache and cache_key in agent_cache:
        return agent_cache[cache_key][0]
    
    # Get agent from database
    agent_db = db.query(Agent).filter(Agent.id == agent_id).first()
    if not agent_db:
        raise ValueError(f"Agent with ID {agent_id} not found")
    
    # Create agent memory
    memory = Memory()
    
    # Create agent instance
    agent_instance = AgentFactory.create_agent(
        agent_type=agent_db.agent_type,
        agent_id=str(agent_db.id),
        llm_service=llm_service,
        memory=memory,
        tool_registry=tool_registry,
        config=agent_db.config
    )
    
    # Cache the instance
    from datetime import datetime
    agent_cache[cache_key] = (agent_instance, datetime.utcnow())
    
    # Simple cache cleanup - if more than 10 agents in cache, remove the oldest ones
    if len(agent_cache) > 10:
        oldest_keys = sorted(
            agent_cache.keys(), 
            key=lambda k: agent_cache[k][1]
        )[:len(agent_cache) - 10]
        
        for key in oldest_keys:
            del agent_cache[key]
    
    return agent_instance

def clear_agent_cache():
    """Clear the agent instance cache completely."""
    global agent_cache
    agent_cache = {}

def get_or_create_conversation_memory(conversation_id: UUID, limit: int = 20) -> Memory:
    """
    Get or create a memory instance for a conversation, pre-loaded with history.
    
    Args:
        conversation_id: Conversation ID
        limit: Maximum number of messages to load
        
    Returns:
        Memory instance
    """
    # Create a new database session
    db = SessionLocal()
    try:
        from arivu.db.models import Message
        
        # Get the conversation history
        messages = db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).order_by(Message.timestamp).limit(limit).all()
        
        # Create memory instance
        memory = Memory()
        
        # Load messages into memory
        user_message = None
        for msg in messages:
            if msg.role == "user":
                user_message = msg.content
            elif msg.role == "agent" and user_message:
                memory.add_interaction(user_message, msg.content)
                user_message = None
        
        return memory
    finally:
        db.close()