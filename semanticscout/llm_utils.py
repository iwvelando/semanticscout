"""
Shared LLM utilities for Semantic Scout.

Provides common retry logic, service restart functionality, and a centralized
LLM client for all LLM interactions via Ollama.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Callable, TypeVar, Optional

import httpx

from .config import LLMConfig

logger = logging.getLogger(__name__)

T = TypeVar('T')

# Global restart coordination to prevent concurrent requests during restart
_restart_lock = asyncio.Lock()
_restart_complete_event: Optional[asyncio.Event] = None


@dataclass
class LLMResponse:
    """Result of an LLM API call."""
    success: bool
    response_text: str
    raw_response: dict
    error: Optional[str] = None


class LLMClient:
    """
    Centralized client for LLM interactions via Ollama.
    
    Handles URL normalization, HTTP client management, error handling,
    and automatic retry with service restart on failures.
    """
    
    def __init__(self, llm_config: LLMConfig):
        """
        Initialize the LLM client.
        
        Args:
            llm_config: LLM configuration.
        """
        self.config = llm_config
        # Normalize base URL - remove trailing slash to prevent double-slash issues
        self.base_url = llm_config.base_url.rstrip('/')
        self.model = llm_config.model
        
        # HTTP client with configurable timeout
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(llm_config.timeout, connect=10.0)
        )
        
        logger.debug(f"LLM client initialized: base_url={self.base_url}, model={self.model}")
    
    async def check_available(self) -> bool:
        """
        Check if Ollama is available and responding.
        
        Returns:
            True if Ollama is available.
        """
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama availability check failed: {e}")
            return False
    
    async def check_model_available(self) -> bool:
        """
        Check if the configured model is available in Ollama.
        
        Returns:
            True if the model is available.
        """
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                return False
            
            data = response.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            
            # Check for exact match or model without tag
            model_base = self.model.split(":")[0]
            for m in models:
                if m == self.model or m.startswith(model_base):
                    return True
            
            logger.warning(f"Model '{self.model}' not found. Available models: {models}")
            return False
            
        except Exception as e:
            logger.error(f"Model availability check failed: {e}")
            return False
    
    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        context_window: Optional[int] = None,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM.
            temperature: Override the default temperature.
            max_tokens: Override the default max tokens (num_predict).
            context_window: Override the default context window (num_ctx).
        
        Returns:
            LLMResponse with the result.
        """
        options = {
            "temperature": temperature if temperature is not None else self.config.temperature,
            "num_predict": max_tokens if max_tokens is not None else self.config.max_tokens,
        }
        
        # Only include context_window if specified or configured
        ctx = context_window if context_window is not None else self.config.context_window
        if ctx:
            options["num_ctx"] = ctx
        
        try:
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": options
                }
            )
            
            if response.status_code != 200:
                error_msg = f"Ollama API error: {response.status_code}"
                logger.error(error_msg)
                return LLMResponse(
                    success=False,
                    response_text="",
                    raw_response={},
                    error=error_msg
                )
            
            result = response.json()
            response_text = result.get("response", "")
            
            return LLMResponse(
                success=True,
                response_text=response_text,
                raw_response=result
            )
            
        except httpx.TimeoutException:
            error_msg = "Ollama request timed out"
            logger.error(error_msg)
            return LLMResponse(
                success=False,
                response_text="",
                raw_response={},
                error=error_msg
            )
        except httpx.ConnectError:
            error_msg = f"Could not connect to Ollama at {self.base_url}"
            logger.error(error_msg)
            return LLMResponse(
                success=False,
                response_text="",
                raw_response={},
                error=error_msg
            )
        except Exception as e:
            error_msg = f"Unexpected error during LLM request: {e}"
            logger.error(error_msg)
            return LLMResponse(
                success=False,
                response_text="",
                raw_response={},
                error=error_msg
            )
    
    async def generate_with_retry(
        self,
        prompt: str,
        operation_name: str = "LLM request",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        context_window: Optional[int] = None,
    ) -> LLMResponse:
        """
        Generate a response with automatic retry and service restart on failure.
        
        Args:
            prompt: The prompt to send to the LLM.
            operation_name: Human-readable name for logging.
            temperature: Override the default temperature.
            max_tokens: Override the default max tokens.
            context_window: Override the default context window.
        
        Returns:
            LLMResponse with the result.
        """
        return await retry_with_restart(
            operation=lambda: self.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                context_window=context_window,
            ),
            llm_config=self.config,
            operation_name=operation_name
        )
    
    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()


async def restart_llm_service(llm_config: LLMConfig) -> bool:
    """
    Restart LLM service by making a configurable HTTP request.
    
    This function coordinates restarts globally so that:
    1. Only one restart is triggered even if multiple tasks fail simultaneously
    2. All concurrent tasks wait for the restart to complete before retrying
    
    Args:
        llm_config: LLM configuration containing restart parameters.
    
    Returns:
        True if restart request succeeded (or was already in progress), False otherwise.
    """
    global _restart_complete_event
    
    if not llm_config.restart_url:
        logger.warning("LLM service restart requested but no restart_url configured")
        return False
    
    async with _restart_lock:
        # Check if another task is already handling the restart
        if _restart_complete_event is not None and not _restart_complete_event.is_set():
            logger.debug("Restart already in progress by another task, will wait for completion")
            event = _restart_complete_event
            # Release lock and wait
        else:
            # We're the first to request a restart - initiate it
            _restart_complete_event = asyncio.Event()
            event = _restart_complete_event
            
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
                        logger.info(f"LLM service restart initiated, blocking all LLM requests for {llm_config.restart_wait_seconds}s")
                        # Perform the wait while holding the lock is fine since
                        # all tasks will wait on the event anyway
                        await asyncio.sleep(llm_config.restart_wait_seconds)
                        logger.info("LLM service restart wait complete, resuming requests")
                        event.set()
                        return True
                    else:
                        logger.error(f"LLM service restart failed: {response.status_code} - {response.text}")
                        event.set()  # Unblock waiters even on failure
                        return False
                        
            except Exception as e:
                logger.error(f"Error requesting LLM service restart: {e}")
                event.set()  # Unblock waiters even on failure
                return False
    
    # If we get here, we're waiting for another task's restart to complete
    logger.debug("Waiting for in-progress LLM restart to complete")
    await event.wait()
    logger.debug("LLM restart complete, proceeding")
    return True


async def wait_for_restart_if_needed() -> None:
    """
    Wait for any in-progress restart to complete before making a request.
    
    This should be called before making LLM API requests to ensure
    we don't hit the service while it's restarting.
    """
    if _restart_complete_event is not None and not _restart_complete_event.is_set():
        logger.debug("Waiting for in-progress LLM restart to complete before request")
        await _restart_complete_event.wait()
        logger.debug("LLM restart complete, proceeding with request")


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
        # Wait for any in-progress restart to complete before attempting
        await wait_for_restart_if_needed()
        
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
