"""
Shared LLM utilities for Semantic Scout.

Provides common retry logic and service restart functionality for LLM interactions.
"""

import asyncio
import logging
from typing import Callable, TypeVar, Optional

import httpx

from .config import LLMConfig

logger = logging.getLogger(__name__)

T = TypeVar('T')


async def restart_llm_service(llm_config: LLMConfig) -> bool:
    """
    Restart LLM service by making a configurable HTTP request.
    
    Args:
        llm_config: LLM configuration containing restart parameters.
    
    Returns:
        True if restart request succeeded, False otherwise.
    """
    if not llm_config.restart_url:
        logger.warning("LLM service restart requested but no restart_url configured")
        return False
    
    try:
        logger.info(f"Requesting LLM service restart via {llm_config.restart_url}")
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Determine HTTP method
            method = llm_config.restart_method.upper()
            
            # Prepare request kwargs
            kwargs = {}
            if llm_config.restart_payload:
                kwargs['json'] = llm_config.restart_payload
            if llm_config.restart_headers:
                kwargs['headers'] = llm_config.restart_headers
            
            # Make the request
            response = await client.request(
                method,
                llm_config.restart_url,
                **kwargs
            )
            
            if response.status_code in [200, 201, 202, 204]:
                logger.info(f"LLM service restart initiated, waiting {llm_config.restart_wait_seconds}s for stabilization")
                await asyncio.sleep(llm_config.restart_wait_seconds)
                return True
            else:
                logger.error(f"LLM service restart failed: {response.status_code} - {response.text}")
                return False
                
    except Exception as e:
        logger.error(f"Error requesting LLM service restart: {e}")
        return False


async def retry_with_restart(
    operation: Callable[[], T],
    llm_config: LLMConfig,
    operation_name: str = "operation"
) -> T:
    """
    Execute an async operation with automatic retry and service restart on failure.
    
    If configured, will restart the LLM service and retry on failures like
    timeouts or 502 errors.
    
    Args:
        operation: Async callable that returns a result with a 'success' attribute.
        llm_config: LLM configuration with retry parameters.
        operation_name: Human-readable name for logging.
    
    Returns:
        Result from the operation.
    """
    max_attempts = 1 + llm_config.retry_attempts
    
    for attempt in range(max_attempts):
        result = await operation()
        
        # Check if operation succeeded
        if hasattr(result, 'success') and result.success:
            return result
        
        # If this was the last attempt, return the failure
        if attempt == max_attempts - 1:
            logger.warning(
                f"{operation_name} failed after {max_attempts} attempts"
            )
            return result
        
        # Log the failure and prepare for retry
        logger.warning(
            f"{operation_name} failed (attempt {attempt + 1}/{max_attempts})"
        )
        
        # Restart service if configured
        if llm_config.restart_url:
            logger.info(f"Attempting service restart before retry {attempt + 2}/{max_attempts}")
            await restart_llm_service(llm_config)
        else:
            logger.info(f"No restart_url configured, retrying without restart")
            # Brief delay before retry even without restart
            await asyncio.sleep(2)
    
    return result  # Should never reach here, but for completeness
