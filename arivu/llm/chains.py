"""
LangChain integration for the Arivu learning platform.

This module provides utilities for creating LangChain chains and agents
for educational interactions.
"""

from typing import Dict, List, Any, Optional
import os
from langchain.chains import ConversationChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.llms.base import BaseLLM
from langchain.callbacks.base import BaseCallbackManager

from .memory import Memory as ArivuMemory

def create_conversation_chain(
    llm: BaseLLM,
    system_prompt: str,
    memory: Optional[ConversationBufferMemory] = None
) -> ConversationChain:
    """
    Create a simple conversation chain with the given LLM.
    
    Args:
        llm: LangChain LLM instance
        system_prompt: System prompt to use
        memory: Optional LangChain memory instance
        
    Returns:
        ConversationChain instance
    """
    if memory is None:
        memory = ConversationBufferMemory()
    
    prompt_template = f"""
    {system_prompt}
    
    Current conversation:
    {{history}}
    Human: {{input}}
    AI:"""
    
    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=prompt_template
    )
    
    return ConversationChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=True
    )

def create_agent_chain(
    llm: BaseLLM,
    tools: List[Tool],
    system_prompt: str,
    memory: Optional[ConversationBufferMemory] = None,
    callback_manager: Optional[BaseCallbackManager] = None
):
    """
    Create an agent with tools for educational interactions.
    
    Args:
        llm: LangChain LLM instance
        tools: List of LangChain tools
        system_prompt: System prompt to use
        memory: Optional LangChain memory instance
        callback_manager: Optional callback manager for monitoring
        
    Returns:
        Agent instance
    """
    if memory is None:
        memory = ConversationBufferMemory(memory_key="chat_history")
    
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        callback_manager=callback_manager,
        agent_kwargs={
            "system_message": system_prompt
        }
    )

def convert_arivu_memory_to_langchain(
    arivu_memory: ArivuMemory
) -> ConversationBufferMemory:
    """
    Convert Arivu memory to LangChain memory.
    
    Args:
        arivu_memory: Arivu Memory instance
        
    Returns:
        LangChain ConversationBufferMemory instance
    """
    langchain_memory = ConversationBufferMemory()
    
    # Get chat history from Arivu memory
    chat_history = arivu_memory.get_chat_history()
    
    # Add messages to LangChain memory
    for message in chat_history:
        if message["role"] == "user":
            langchain_memory.chat_memory.add_user_message(message["content"])
        elif message["role"] == "assistant":
            langchain_memory.chat_memory.add_ai_message(message["content"])
    
    return langchain_memory

def create_subject_specific_tools(subject: str) -> List[Tool]:
    """
    Create subject-specific tools for agents.
    
    Args:
        subject: Subject area (math, science, language, history)
        
    Returns:
        List of subject-specific tools
    """
    # This would be expanded with actual tool implementations
    tools = []
    
    if subject == "math":
        # Math-specific tools
        pass
    elif subject == "science":
        # Science-specific tools
        pass
    elif subject == "language":
        # Language-specific tools
        pass
    elif subject == "history":
        # History-specific tools
        pass
    
    return tools