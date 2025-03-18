"""
Dependencies for the Arivu learning platform.

This module provides FastAPI dependency functions for various services
and utilities used throughout the application.
"""

from typing import Generator, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
import jwt
from jwt.exceptions import PyJWTError

from arivu.db.database import get_db
from arivu.auth.jwt import decode_access_token
from arivu.users.models import User
from arivu.llm.service import LLMService
from arivu.llm.config import get_llm_service
from arivu.tools.registry import ToolRegistry, get_tool_registry

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """
    Get the current authenticated user.
    
    Args:
        token: JWT token from request
        db: Database session
        
    Returns:
        User model
        
    Raises:
        HTTPException: If authentication fails
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = decode_access_token(token)
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except PyJWTError:
        raise credentials_exception
    
    from arivu.users.service import get_user_by_username
    user = get_user_by_username(db, username=username)
    if user is None:
        raise credentials_exception
        
    return user

async def get_current_admin_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get the current authenticated admin user.
    
    Args:
        current_user: Authenticated user
        
    Returns:
        User model
        
    Raises:
        HTTPException: If user is not an admin
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized. Admin privileges required."
        )
    return current_user

def get_llm_service_dependency() -> LLMService:
    """
    Get the LLM service instance.
    
    Returns:
        LLMService instance
    """
    return get_llm_service()

def get_tool_registry_dependency() -> ToolRegistry:
    """
    Get the tool registry instance.
    
    Returns:
        ToolRegistry instance
    """
    return get_tool_registry()