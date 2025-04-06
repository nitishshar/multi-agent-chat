from cachetools import TTLCache
from functools import wraps
import streamlit as st
import time
from typing import Any, Callable, Dict, Optional, TypeVar, cast

# Type variables for generic functions
T = TypeVar('T')
R = TypeVar('R')

# Constants
CACHE_TTL = 300  # Default TTL in seconds (5 minutes)
CACHE_MAX_ENTRIES = 100  # Default maximum cache entries


def cached(ttl: int = CACHE_TTL, maxsize: int = CACHE_MAX_ENTRIES):
    """
    Custom caching decorator with TTL for functions outside Streamlit.
    
    Args:
        ttl: Time to live in seconds
        maxsize: Maximum number of entries in the cache
        
    Returns:
        Decorated function with caching
    """
    cache = TTLCache(maxsize=maxsize, ttl=ttl)
    
    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            # Create a key from the function name, args, and kwargs
            key_parts = [func.__name__]
            key_parts.extend([str(arg) for arg in args])
            key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
            key = ":".join(key_parts)
            
            # Check if result is in cache
            if key in cache:
                return cache[key]
            
            # Execute the function and cache the result
            result = func(*args, **kwargs)
            cache[key] = result
            return result
        
        # Add cache clearing method
        def cache_clear() -> None:
            cache.clear()
        
        wrapper.cache_clear = cache_clear  # type: ignore
        
        return wrapper
    
    return decorator


class SessionState:
    """
    Utility class for managing Streamlit session state.
    All methods are static to avoid instance creation.
    """
    
    @staticmethod
    def initialize() -> None:
        """
        Initialize session state variables if they don't exist.
        """
        # Chat session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # UI container references
        if "containers" not in st.session_state:
            st.session_state.containers = {
                "title": st.container(),
                "status": st.empty(),
                "error_recovery": st.container(),
                "main_area": st.container(),
            }
        
        # File upload state preservation
        if "file_uploader_key" not in st.session_state:
            st.session_state.file_uploader_key = f"uploader_{int(time.time())}"
    
    @staticmethod
    def get(key: str, default: Optional[Any] = None) -> Any:
        """
        Get a value from session state with a default.
        
        Args:
            key: Session state key
            default: Default value if key doesn't exist
            
        Returns:
            Value from session state or default
        """
        return st.session_state.get(key, default)
    
    @staticmethod
    def set(key: str, value: Any) -> None:
        """
        Set a value in session state.
        
        Args:
            key: Session state key
            value: Value to set
        """
        st.session_state[key] = value
    
    @staticmethod
    def clear_chat_history() -> None:
        """
        Clear chat history from session state.
        """
        st.session_state.chat_history = []
    
    @staticmethod
    def preserve_file_uploader_state() -> None:
        """
        Critical method to ensure file uploader state is preserved across reruns.
        """
        if "file_uploader_key" in st.session_state:
            st.session_state.file_uploader_key = st.session_state.file_uploader_key 