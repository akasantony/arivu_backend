"""
LLM module for the Arivu learning platform.

This module provides services for interacting with language models,
managing conversation memory, and handling prompt templates.
"""

from .service import LLMService
from .memory import Memory
from .prompts import get_prompt_template, register_prompt_template
from .chains import create_agent_chain

__all__ = [
    'LLMService',
    'Memory',
    'get_prompt_template',
    'register_prompt_template',
    'create_agent_chain'
]