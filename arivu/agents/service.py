from typing import List, Optional, Dict, Any, Union
from sqlalchemy.orm import Session
from sqlalchemy import desc
from datetime import datetime
from uuid import UUID
import random

from arivu.db import models as db_models
from arivu.agents.base_agent import BaseAgent
from . import models

def create_agent(db: Session, agent_data: models.AgentCreate) -> db_models.Agent:
    """
    Create a new agent in the database.
    
    Args:
        db: Database session
        agent_data: Agent data to create
        
    Returns:
        Created agent model
    """
    db_agent = db_models.Agent(
        name=agent_data.name,
        description=agent_data.description,
        agent_type=agent_data.agent_type.value,
        default_learning_level=agent_data.default_learning_level.value,
        enabled=agent_data.enabled,
        config=agent_data.config,
    )
    db.add(db_agent)
    db.commit()
    db.refresh(db_agent)
    return db_agent

def get_agents(
    db: Session, 
    enabled: Optional[bool] = None, 
    agent_type: Optional[models.AgentType] = None
) -> List[db_models.Agent]:
    """
    Get a list of agents with optional filtering.
    
    Args:
        db: Database session
        enabled: Filter by enabled status
        agent_type: Filter by agent type
        
    Returns:
        List of agent models
    """
    query = db.query(db_models.Agent)
    
    if enabled is not None:
        query = query.filter(db_models.Agent.enabled == enabled)
    
    if agent_type is not None:
        query = query.filter(db_models.Agent.agent_type == agent_type.value)
    
    return query.all()

def get_agent(db: Session, agent_id: UUID) -> Optional[db_models.Agent]:
    """
    Get a specific agent by ID.
    
    Args:
        db: Database session
        agent_id: Agent ID
        
    Returns:
        Agent model or None if not found
    """
    return db.query(db_models.Agent).filter(db_models.Agent.id == agent_id).first()

def update_agent(
    db: Session, 
    agent: db_models.Agent, 
    agent_data: models.AgentUpdate
) -> db_models.Agent:
    """
    Update an existing agent.
    
    Args:
        db: Database session
        agent: Agent model to update
        agent_data: New agent data
        
    Returns:
        Updated agent model
    """
    update_data = agent_data.dict(exclude_unset=True)
    
    # Convert enum values to strings
    if "default_learning_level" in update_data and update_data["default_learning_level"]:
        update_data["default_learning_level"] = update_data["default_learning_level"].value
    
    for key, value in update_data.items():
        setattr(agent, key, value)
    
    agent.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(agent)
    return agent

def delete_agent(db: Session, agent: db_models.Agent) -> None:
    """
    Delete an agent.
    
    Args:
        db: Database session
        agent: Agent model to delete
    """
    db.delete(agent)
    db.commit()

def create_conversation(
    db: Session, 
    conversation_data: models.ConversationCreate
) -> db_models.Conversation:
    """
    Create a new conversation.
    
    Args:
        db: Database session
        conversation_data: Conversation data to create
        
    Returns:
        Created conversation model
    """
    db_conversation = db_models.Conversation(
        title=conversation_data.title,
        user_id=conversation_data.user_id,
        agent_id=conversation_data.agent_id,
        learning_level=conversation_data.learning_level.value,
        metadata=conversation_data.metadata,
    )
    db.add(db_conversation)
    db.commit()
    db.refresh(db_conversation)
    
    # Add a system message to start the conversation
    agent = get_agent(db, conversation_data.agent_id)
    system_message = (
        f"Welcome to a conversation with {agent.name}, your {agent.description} tutor. "
        f"Learning level is set to {conversation_data.learning_level.value.replace('_', ' ')}. "
        f"Feel free to ask questions about any topics in {agent.description}!"
    )
    
    add_message(
        db=db,
        conversation_id=db_conversation.id,
        content=system_message,
        role=models.MessageRole.SYSTEM,
        metadata={"event_type": "conversation_start"}
    )
    
    return db_conversation

def get_conversation(
    db: Session, 
    conversation_id: UUID
) -> Optional[db_models.Conversation]:
    """
    Get a specific conversation by ID.
    
    Args:
        db: Database session
        conversation_id: Conversation ID
        
    Returns:
        Conversation model or None if not found
    """
    return db.query(db_models.Conversation).filter(
        db_models.Conversation.id == conversation_id
    ).first()

def get_conversation_details(
    db: Session, 
    conversation: db_models.Conversation
) -> models.ConversationDetails:
    """
    Get detailed conversation information including agent details.
    
    Args:
        db: Database session
        conversation: Conversation model
        
    Returns:
        Conversation details model
    """
    agent = get_agent(db, conversation.agent_id)
    
    # Get recent messages (limited to keep response size reasonable)
    messages = db.query(db_models.Message).filter(
        db_models.Message.conversation_id == conversation.id
    ).order_by(db_models.Message.timestamp.desc()).limit(10).all()
    
    # Reverse to get chronological order
    messages.reverse()
    
    # Convert to Pydantic model
    conversation_model = models.Conversation.from_orm(conversation)
    conversation_model.messages = [models.Message.from_orm(msg) for msg in messages]
    
    # Add agent details
    agent_model = models.Agent.from_orm(agent)
    
    return models.ConversationDetails(
        **conversation_model.dict(),
        agent=agent_model
    )

def list_conversations(
    db: Session,
    user_id: Optional[UUID] = None,
    agent_id: Optional[UUID] = None,
    limit: int = 50,
    offset: int = 0
) -> List[db_models.Conversation]:
    """
    List conversations with optional filtering.
    
    Args:
        db: Database session
        user_id: Filter by user ID
        agent_id: Filter by agent ID
        limit: Maximum number of results
        offset: Offset for pagination
        
    Returns:
        List of conversation models
    """
    query = db.query(db_models.Conversation)
    
    if user_id is not None:
        query = query.filter(db_models.Conversation.user_id == user_id)
    
    if agent_id is not None:
        query = query.filter(db_models.Conversation.agent_id == agent_id)
    
    query = query.order_by(desc(db_models.Conversation.updated_at))
    
    return query.offset(offset).limit(limit).all()

def update_conversation(
    db: Session,
    conversation: db_models.Conversation,
    conversation_data: models.ConversationUpdate
) -> db_models.Conversation:
    """
    Update an existing conversation.
    
    Args:
        db: Database session
        conversation: Conversation model to update
        conversation_data: New conversation data
        
    Returns:
        Updated conversation model
    """
    update_data = conversation_data.dict(exclude_unset=True)
    
    # Convert enum values to strings
    if "learning_level" in update_data and update_data["learning_level"]:
        update_data["learning_level"] = update_data["learning_level"].value
    
    for key, value in update_data.items():
        setattr(conversation, key, value)
    
    conversation.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(conversation)
    
    # If learning level was updated, add a system message about it
    if conversation_data.learning_level:
        add_message(
            db=db,
            conversation_id=conversation.id,
            content=f"Learning level changed to {conversation_data.learning_level.value.replace('_', ' ')}.",
            role=models.MessageRole.SYSTEM,
            metadata={"event_type": "learning_level_change"}
        )
    
    return conversation

def delete_conversation(
    db: Session, 
    conversation: db_models.Conversation
) -> None:
    """
    Delete a conversation.
    
    Args:
        db: Database session
        conversation: Conversation model to delete
    """
    # Delete all messages in the conversation first
    db.query(db_models.Message).filter(
        db_models.Message.conversation_id == conversation.id
    ).delete()
    
    # Then delete the conversation
    db.delete(conversation)
    db.commit()

def add_message(
    db: Session,
    conversation_id: UUID,
    content: str,
    role: models.MessageRole,
    metadata: Optional[Dict[str, Any]] = None
) -> db_models.Message:
    """
    Add a message to a conversation.
    
    Args:
        db: Database session
        conversation_id: Conversation ID
        content: Message content
        role: Message role (user, agent, system)
        metadata: Optional message metadata
        
    Returns:
        Created message model
    """
    db_message = db_models.Message(
        conversation_id=conversation_id,
        content=content,
        role=role.value,
        metadata=metadata or {},
    )
    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    
    # Update conversation timestamp
    conversation = get_conversation(db, conversation_id)
    conversation.updated_at = datetime.utcnow()
    db.commit()
    
    return db_message

def get_conversation_messages(
    db: Session,
    conversation_id: UUID,
    limit: int = 50,
    offset: int = 0
) -> List[db_models.Message]:
    """
    Get messages for a conversation.
    
    Args:
        db: Database session
        conversation_id: Conversation ID
        limit: Maximum number of messages to return
        offset: Offset for pagination
        
    Returns:
        List of message models
    """
    return db.query(db_models.Message).filter(
        db_models.Message.conversation_id == conversation_id
    ).order_by(db_models.Message.timestamp).offset(offset).limit(limit).all()

def update_conversation_learning_metrics(
    db: Session,
    conversation_id: UUID,
    agent_instance: BaseAgent
) -> None:
    """
    Update conversation metadata with latest learning metrics.
    
    Args:
        db: Database session
        conversation_id: Conversation ID
        agent_instance: Agent instance with updated metrics
    """
    conversation = get_conversation(db, conversation_id)
    if not conversation:
        return
    
    # Get strengths and weaknesses analysis
    analysis = agent_instance.analyze_strengths_weaknesses()
    
    # Update conversation metadata with latest metrics
    if not conversation.metadata:
        conversation.metadata = {}
    
    conversation.metadata.update({
        "learning_metrics": {
            "strengths": analysis["strengths"],
            "weaknesses": analysis["weaknesses"],
            "learning_pace": analysis["learning_pace"],
            "recommended_focus_areas": analysis["recommended_focus_areas"],
            "current_level": agent_instance.current_level.id,
            "last_updated": datetime.utcnow().isoformat()
        }
    })
    
    db.commit()

def generate_suggested_questions(
    agent_instance: BaseAgent,
    conversation_history: List[db_models.Message],
    agent_response: Any
) -> List[str]:
    """
    Generate suggested follow-up questions based on the conversation.
    
    Args:
        agent_instance: Agent instance
        conversation_history: Conversation history
        agent_response: Agent's response data
        
    Returns:
        List of suggested questions
    """
    # This is a simplified implementation
    # In a real system, this might call the LLM to generate contextually relevant questions
    
    # Get the agent type to suggest type-specific questions
    agent_type = agent_instance.__class__.__name__
    
    suggested_questions = []
    
    if agent_type == "MathAgent":
        # Math-specific suggestions
        suggested_questions = [
            "Can you explain that step-by-step?",
            "What's a real-world application of this concept?",
            "Can you give me a simpler example?",
            "How does this relate to [previous topic]?",
            "Can you create a practice problem for me to solve?"
        ]
    elif agent_type == "ScienceAgent":
        # Science-specific suggestions
        suggested_questions = [
            "Why does this phenomenon occur?",
            "Can you explain the underlying principles?",
            "How was this discovery made historically?",
            "What are some real-world applications of this?",
            "Can you suggest an experiment to demonstrate this?"
        ]
    elif agent_type == "LanguageAgent":
        # Language-specific suggestions
        suggested_questions = [
            "Can you give me an example sentence?",
            "What's the difference between this and [similar concept]?",
            "How would I use this in conversation?",
            "Can you suggest some practice exercises?",
            "What are common mistakes people make with this?"
        ]
    elif agent_type == "HistoryAgent":
        # History-specific suggestions
        suggested_questions = [
            "How did this affect later historical events?",
            "What were the different perspectives on this event?",
            "How does this connect to modern issues?",
            "What primary sources document this event?",
            "How have historians' views on this changed over time?"
        ]
    else:
        # Generic suggestions
        suggested_questions = [
            "Can you explain that in more detail?",
            "How does this relate to what we discussed earlier?",
            "Can you give me an example?",
            "Why is this important to understand?",
            "What should I learn about next?"
        ]
    
    # Randomly select 3 questions to avoid repetition
    if len(suggested_questions) > 3:
        suggested_questions = random.sample(suggested_questions, 3)
    
    return suggested_questions