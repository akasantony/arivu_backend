"""
Configuration for the LLM module in the Arivu learning platform.
"""

import os
from typing import Dict, Any, Optional
from functools import lru_cache

from pydantic import BaseSettings, Field

from .service import LLMProvider, LLMConfig, LLMService

class LLMSettings(BaseSettings):
    """
    Settings for the LLM service.
    
    Uses environment variables with ARIVU_LLM_ prefix.
    """
    provider: str = Field(default="openai", env="ARIVU_LLM_PROVIDER")
    model_name: str = Field(default="gpt-3.5-turbo", env="ARIVU_LLM_MODEL_NAME")
    api_key: Optional[str] = Field(default=None, env="ARIVU_LLM_API_KEY")
    api_base: Optional[str] = Field(default=None, env="ARIVU_LLM_API_BASE")
    temperature: float = Field(default=0.7, env="ARIVU_LLM_TEMPERATURE")
    max_tokens: Optional[int] = Field(default=None, env="ARIVU_LLM_MAX_TOKENS")
    timeout: int = Field(default=30, env="ARIVU_LLM_TIMEOUT")
    
    # Provider-specific settings
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    cohere_api_key: Optional[str] = Field(default=None, env="COHERE_API_KEY")
    huggingface_api_key: Optional[str] = Field(default=None, env="HUGGINGFACE_API_KEY")
    azure_openai_api_key: Optional[str] = Field(default=None, env="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: Optional[str] = Field(default=None, env="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_version: str = Field(default="2023-05-15", env="AZURE_OPENAI_API_VERSION")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_llm_settings() -> LLMSettings:
    """
    Get LLM settings from environment variables.
    
    Returns:
        LLMSettings instance
    """
    return LLMSettings()

def create_llm_config(settings: Optional[LLMSettings] = None) -> LLMConfig:
    """
    Create an LLM configuration from settings.
    
    Args:
        settings: Optional settings (uses environment variables if not provided)
        
    Returns:
        LLMConfig instance
    """
    if settings is None:
        settings = get_llm_settings()
    
    # Determine provider enum value
    provider_str = settings.provider.lower()
    try:
        provider = LLMProvider(provider_str)
    except ValueError:
        provider = LLMProvider.OPENAI
    
    # Get the appropriate API key based on provider
    api_key = settings.api_key
    if not api_key:
        if provider == LLMProvider.OPENAI:
            api_key = settings.openai_api_key
        elif provider == LLMProvider.ANTHROPIC:
            api_key = settings.anthropic_api_key
        elif provider == LLMProvider.COHERE:
            api_key = settings.cohere_api_key
        elif provider == LLMProvider.HUGGINGFACE:
            api_key = settings.huggingface_api_key
        elif provider == LLMProvider.AZURE_OPENAI:
            api_key = settings.azure_openai_api_key
    
    # Get the appropriate API base based on provider
    api_base = settings.api_base
    if not api_base and provider == LLMProvider.AZURE_OPENAI:
        api_base = settings.azure_openai_endpoint
    
    # Additional parameters based on provider
    additional_params = {}
    if provider == LLMProvider.AZURE_OPENAI:
        additional_params["api_version"] = settings.azure_openai_api_version
    
    return LLMConfig(
        provider=provider,
        model_name=settings.model_name,
        api_key=api_key,
        api_base=api_base,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        timeout=settings.timeout,
        additional_params=additional_params
    )

@lru_cache()
def get_llm_service() -> LLMService:
    """
    Create an LLM service singleton.
    
    Returns:
        LLMService instance
    """
    config = create_llm_config()
    return LLMService(config=config)