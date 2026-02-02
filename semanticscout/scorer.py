"""
Semantic Scorer module for Semantic Scout.

Uses a local LLM via Ollama to score job listings based on relevance
to the user's job goal statement.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

import httpx

from .config import LLMConfig, SearchConfig
from .database import Job

logger = logging.getLogger(__name__)


@dataclass
class ScoringResult:
    """Result of scoring a job listing."""
    score: float
    reasoning: str
    raw_response: str
    success: bool
    error: Optional[str] = None


class SemanticScorer:
    """
    Scores job listings using a local LLM via Ollama.
    
    Sends job descriptions to the LLM and receives a relevance score
    based on the configured job goal statement.
    """
    
    def __init__(self, llm_config: LLMConfig, search_config: SearchConfig):
        """
        Initialize the scorer.
        
        Args:
            llm_config: LLM configuration.
            search_config: Search configuration containing the job goal statement.
        """
        self.llm_config = llm_config
        self.job_goal = search_config.job_goal_statement
        self.base_url = llm_config.base_url.rstrip('/')
        self.model = llm_config.model
        
        # HTTP client with timeout
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(llm_config.timeout, connect=10.0)
        )
        
        logger.info(f"Semantic scorer initialized with model: {self.model}")
    
    def _build_scoring_prompt(self, job: Job) -> str:
        """
        Build the prompt for scoring a job.
        
        Args:
            job: The job to score.
        
        Returns:
            Formatted prompt string.
        """
        prompt = f"""You are an expert job matching assistant. Your task is to score how well a job listing matches a candidate's career goals.

## Candidate's Job Goal Statement:
{self.job_goal}

## Job Listing to Evaluate:

**Title:** {job.title}
**Company:** {job.company}
**Location:** {job.location}

**Job Description:**
{job.description[:4000]}  

## Scoring Instructions:

Score this job from 1 to 10 based on how well it matches the candidate's goals:

- **9-10:** Exceptional match - Role perfectly aligns with goals, requires the exact skills mentioned, and offers the desired level of responsibility.
- **7-8:** Strong match - Role closely aligns with goals, requires most of the key skills, and is likely a good fit.
- **5-6:** Moderate match - Role has some alignment, but may lack key technical requirements or desired responsibilities.
- **3-4:** Weak match - Role has minimal alignment with goals or requires significantly different skills.
- **1-2:** Poor match - Role does not match the candidate's goals or skill set.

## Required Output Format:

Respond ONLY with a JSON object in this exact format:
{{
    "score": <number between 1 and 10>,
    "reasoning": "<2-3 sentences explaining the score, highlighting matching skills/requirements and any gaps>"
}}

Do not include any other text outside the JSON object."""

        return prompt
    
    async def check_ollama_available(self) -> bool:
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
    
    async def _restart_ollama(self) -> bool:
        """
        Restart Ollama service by POSTing to the configured restart URL.
        
        Returns:
            True if restart request succeeded, False otherwise.
        """
        if not self.llm_config.restart_url:
            logger.warning("Ollama restart requested but no restart_url configured")
            return False
        
        try:
            logger.info(f"Requesting Ollama restart via {self.llm_config.restart_url}")
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.llm_config.restart_url,
                    json={"action": "restart", "mode": "replace"},
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code in [200, 201, 202, 204]:
                    logger.info(f"Ollama restart initiated, waiting {self.llm_config.restart_wait_seconds}s for stabilization")
                    await asyncio.sleep(self.llm_config.restart_wait_seconds)
                    return True
                else:
                    logger.error(f"Ollama restart failed: {response.status_code} - {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error requesting Ollama restart: {e}")
            return False
    
    async def score_job(self, job: Job) -> ScoringResult:
        """
        Score a single job listing with automatic retry on failure.
        
        If configured, will restart Ollama and retry on failures like
        timeouts or 502 errors.
        
        Args:
            job: The job to score.
        
        Returns:
            ScoringResult with the score and reasoning.
        """
        max_attempts = 1 + self.llm_config.retry_attempts
        
        for attempt in range(max_attempts):
            result = await self._attempt_score_job(job)
            
            if result.success:
                return result
            
            # If this was the last attempt, return the failure
            if attempt == max_attempts - 1:
                logger.warning(
                    f"Job '{job.title}' scoring failed after {max_attempts} attempts: {result.error}"
                )
                return result
            
            # Restart Ollama and retry
            logger.warning(
                f"Job '{job.title}' scoring failed (attempt {attempt + 1}/{max_attempts}): {result.error}"
            )
            
            if self.llm_config.restart_url:
                logger.info(f"Attempting Ollama restart before retry {attempt + 2}/{max_attempts}")
                await self._restart_ollama()
            else:
                logger.info(f"No restart_url configured, retrying without restart")
                # Brief delay before retry even without restart
                await asyncio.sleep(2)
        
        return result  # Should never reach here, but for completeness
    
    async def _attempt_score_job(self, job: Job) -> ScoringResult:
        """
        Single attempt to score a job (internal method).
        
        Args:
            job: The job to score.
        
        Returns:
            ScoringResult with the score and reasoning.
        """
        prompt = self._build_scoring_prompt(job)
        
        try:
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.llm_config.temperature,
                        "num_predict": self.llm_config.max_tokens,
                        "num_ctx": self.llm_config.context_window,
                    }
                }
            )
            
            if response.status_code != 200:
                error_msg = f"Ollama API error: {response.status_code}"
                logger.error(error_msg)
                return ScoringResult(
                    score=0,
                    reasoning="",
                    raw_response=response.text,
                    success=False,
                    error=error_msg
                )
            
            result = response.json()
            raw_response = result.get("response", "")
            
            # Parse the JSON response
            return self._parse_scoring_response(raw_response)
            
        except httpx.TimeoutException:
            error_msg = "Ollama request timed out"
            logger.error(error_msg)
            return ScoringResult(
                score=0,
                reasoning="",
                raw_response="",
                success=False,
                error=error_msg
            )
        except httpx.ConnectError:
            error_msg = f"Could not connect to Ollama at {self.base_url}"
            logger.error(error_msg)
            return ScoringResult(
                score=0,
                reasoning="",
                raw_response="",
                success=False,
                error=error_msg
            )
        except Exception as e:
            error_msg = f"Unexpected error during scoring: {e}"
            logger.error(error_msg)
            return ScoringResult(
                score=0,
                reasoning="",
                raw_response="",
                success=False,
                error=error_msg
            )
    
    def _parse_scoring_response(self, response: str) -> ScoringResult:
        """
        Parse the LLM's response to extract score and reasoning.
        
        Args:
            response: Raw response from the LLM.
        
        Returns:
            Parsed ScoringResult.
        """
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                
                score = float(data.get("score", 0))
                reasoning = data.get("reasoning", "")
                
                # Validate score range
                if not 1 <= score <= 10:
                    score = max(1, min(10, score))
                    logger.warning(f"Score was outside 1-10 range, clamped to {score}")
                
                return ScoringResult(
                    score=score,
                    reasoning=reasoning,
                    raw_response=response,
                    success=True
                )
            
            # Try to extract score with regex fallback
            score_match = re.search(r'(?:score|rating)[:\s]*(\d+(?:\.\d+)?)', response, re.IGNORECASE)
            if score_match:
                score = float(score_match.group(1))
                score = max(1, min(10, score))
                
                return ScoringResult(
                    score=score,
                    reasoning=response[:500],
                    raw_response=response,
                    success=True
                )
            
            # Could not parse
            logger.warning(f"Could not parse scoring response: {response[:200]}")
            return ScoringResult(
                score=0,
                reasoning="",
                raw_response=response,
                success=False,
                error="Could not parse LLM response"
            )
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            return ScoringResult(
                score=0,
                reasoning="",
                raw_response=response,
                success=False,
                error=f"JSON parse error: {e}"
            )
        except Exception as e:
            logger.error(f"Response parsing error: {e}")
            return ScoringResult(
                score=0,
                reasoning="",
                raw_response=response,
                success=False,
                error=str(e)
            )
    
    async def score_jobs(
        self, 
        jobs: list[Job], 
        max_concurrent: int = 2,
        delay_between: float = 1.0
    ) -> list[tuple[Job, ScoringResult]]:
        """
        Score multiple jobs with rate limiting.
        
        Args:
            jobs: List of jobs to score.
            max_concurrent: Maximum concurrent scoring requests.
            delay_between: Delay in seconds between batches.
        
        Returns:
            List of (job, result) tuples.
        """
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def score_with_limit(job: Job) -> tuple[Job, ScoringResult]:
            async with semaphore:
                result = await self.score_job(job)
                await asyncio.sleep(delay_between)
                return job, result
        
        tasks = [score_with_limit(job) for job in jobs]
        
        for i, task in enumerate(asyncio.as_completed(tasks)):
            job, result = await task
            results.append((job, result))
            
            if result.success:
                logger.info(
                    f"[{i+1}/{len(jobs)}] Scored '{job.title}' at {job.company}: {result.score}/10"
                )
            else:
                logger.warning(
                    f"[{i+1}/{len(jobs)}] Failed to score '{job.title}': {result.error}"
                )
        
        return results
    
    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()


class ScorerManager:
    """Manages the scoring lifecycle."""
    
    def __init__(self, llm_config: LLMConfig, search_config: SearchConfig):
        """
        Initialize the scorer manager.
        
        Args:
            llm_config: LLM configuration.
            search_config: Search configuration.
        """
        self.scorer = SemanticScorer(llm_config, search_config)
        self.model = llm_config.model
    
    async def initialize(self) -> bool:
        """
        Initialize and verify Ollama connectivity.
        
        Returns:
            True if Ollama is available and model is ready.
        """
        # Check Ollama is running
        if not await self.scorer.check_ollama_available():
            logger.error("Ollama is not available. Please start Ollama first.")
            return False
        
        # Check model is available
        if not await self.scorer.check_model_available():
            logger.warning(
                f"Model '{self.model}' not found. You may need to pull it with: "
                f"ollama pull {self.model}"
            )
            # Don't fail - the model might be pulled on first use
        
        logger.info("Scorer manager initialized successfully")
        return True
    
    async def score_jobs(
        self, 
        jobs: list[Job],
        max_concurrent: int = 2
    ) -> list[tuple[Job, ScoringResult]]:
        """
        Score a list of jobs.
        
        Args:
            jobs: Jobs to score.
            max_concurrent: Maximum concurrent requests.
        
        Returns:
            List of (job, result) tuples.
        """
        return await self.scorer.score_jobs(jobs, max_concurrent=max_concurrent)
    
    async def close(self) -> None:
        """Clean up resources."""
        await self.scorer.close()
