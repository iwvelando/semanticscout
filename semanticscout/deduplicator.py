"""
Embedding-based job deduplication for Semantic Scout.

Uses sentence-transformers to create embeddings of job signatures
(title | company | location) and detects duplicates across sources
using cosine similarity.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Lazy load sentence-transformers to avoid import overhead when not needed
_model = None
_device = None


def _get_model():
    """Lazy load the sentence transformer model with GPU fallback to CPU."""
    global _model, _device
    if _model is None:
        import torch
        from sentence_transformers import SentenceTransformer
        
        # Check GPU compatibility
        use_cuda = False
        if torch.cuda.is_available():
            try:
                # Try to get GPU info
                gpu_name = torch.cuda.get_device_name(0)
                capability = torch.cuda.get_device_capability(0)
                logger.info(f"Found GPU: {gpu_name} (compute capability {capability[0]}.{capability[1]})")
                
                # PyTorch 2.x requires compute capability >= 7.0
                # GTX 1080/1070/1060 etc are 6.x (Pascal) - not supported
                if capability[0] >= 7:
                    use_cuda = True
                    _device = "cuda"
                    logger.info("GPU compute capability sufficient, using CUDA")
                else:
                    logger.warning(
                        f"GPU compute capability {capability[0]}.{capability[1]} < 7.0 required by PyTorch. "
                        f"Falling back to CPU for embeddings (still fast for this model)."
                    )
                    _device = "cpu"
            except Exception as e:
                logger.warning(f"GPU check failed: {e}. Using CPU.")
                _device = "cpu"
        else:
            logger.info("No CUDA GPU available, using CPU")
            _device = "cpu"
        
        logger.info(f"Loading sentence-transformers model: all-MiniLM-L6-v2 (device={_device})")
        _model = SentenceTransformer('all-MiniLM-L6-v2', device=_device)
        logger.info("Sentence-transformers model loaded successfully")
    return _model
    return _model


@dataclass
class DuplicateMatch:
    """Result of a duplicate detection check."""
    is_duplicate: bool
    matched_job_id: Optional[int]
    matched_url: Optional[str]
    similarity_score: float
    new_signature: str
    matched_signature: Optional[str]


class JobDeduplicator:
    """
    Embedding-based job deduplicator.
    
    Uses cosine similarity of job signature embeddings to detect
    duplicates across different job board sources.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.92,
        use_gpu: bool = True
    ):
        """
        Initialize the deduplicator.
        
        Args:
            similarity_threshold: Minimum cosine similarity to consider jobs duplicates.
                                  0.92 is a good balance of precision/recall.
            use_gpu: Whether to use GPU for embedding computation.
        """
        self.similarity_threshold = similarity_threshold
        self.use_gpu = use_gpu
        self._embeddings_cache: dict[int, np.ndarray] = {}
        self._signatures_cache: dict[int, str] = {}
        self._model_loaded = False
    
    def _ensure_model_loaded(self) -> None:
        """Ensure the model is loaded (lazy loading)."""
        if not self._model_loaded:
            _get_model()
            self._model_loaded = True
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text for better matching.
        
        Handles common variations like:
        - Sr. vs Senior
        - Corp vs Corporation
        - Extra whitespace
        """
        if not text:
            return ""
        
        text = text.lower().strip()
        
        # Common abbreviation expansions
        replacements = {
            " sr. ": " senior ",
            " sr ": " senior ",
            " jr. ": " junior ",
            " jr ": " junior ",
            " corp.": " corporation",
            " corp ": " corporation ",
            " inc.": " incorporated",
            " inc ": " incorporated ",
            " llc": "",
            " ltd": "",
            " l.l.c.": "",
            " & ": " and ",
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Normalize whitespace
        text = " ".join(text.split())
        
        return text
    
    def get_job_signature(self, title: str, company: str, location: str, description: str = "") -> str:
        """
        Create a normalized text signature for embedding.
        
        The signature combines title, company, location, and a snippet of
        the description to improve duplicate detection accuracy.
        
        Args:
            title: Job title.
            company: Company name.
            location: Job location.
            description: Job description (first ~200 chars used).
        
        Returns:
            Normalized signature string.
        """
        norm_title = self.normalize_text(title)
        norm_company = self.normalize_text(company)
        norm_location = self.normalize_text(location)
        
        # Include first ~200 chars of description for better discrimination
        desc_snippet = ""
        if description:
            desc_snippet = self.normalize_text(description[:200])
        
        # Format: title | company | location | description_snippet
        signature = f"{norm_title} | {norm_company} | {norm_location}"
        if desc_snippet:
            signature += f" | {desc_snippet}"
        
        return signature
    
    def compute_embedding(self, signature: str) -> np.ndarray:
        """
        Compute embedding for a job signature.
        
        Args:
            signature: The job signature string.
        
        Returns:
            Embedding vector as numpy array.
        """
        self._ensure_model_loaded()
        model = _get_model()
        
        # Normalize the embedding for cosine similarity
        # show_progress_bar=False suppresses the "Batches: 100%" output
        embedding = model.encode(signature, normalize_embeddings=True, show_progress_bar=False)
        return embedding
    
    def check_duplicate(
        self,
        new_title: str,
        new_company: str,
        new_location: str,
        new_description: str,
        new_url: str,
        existing_jobs: list[tuple[int, str, str, str, str, str]]  # (id, title, company, location, description, url)
    ) -> DuplicateMatch:
        """
        Check if a new job is a duplicate of any existing jobs.
        
        Args:
            new_title: Title of the new job.
            new_company: Company of the new job.
            new_location: Location of the new job.
            new_description: Description of the new job.
            new_url: URL of the new job.
            existing_jobs: List of existing jobs as (id, title, company, location, description, url) tuples.
        
        Returns:
            DuplicateMatch with match details.
        """
        if not existing_jobs:
            return DuplicateMatch(
                is_duplicate=False,
                matched_job_id=None,
                matched_url=None,
                similarity_score=0.0,
                new_signature=self.get_job_signature(new_title, new_company, new_location, new_description),
                matched_signature=None
            )
        
        new_signature = self.get_job_signature(new_title, new_company, new_location, new_description)
        new_embedding = self.compute_embedding(new_signature)
        
        # If critical fields are missing, use a much stricter threshold to avoid false positives
        # When company is empty, jobs with same title would otherwise always match
        # We need BOTH company AND either location or substantial description to trust dedup
        has_company = bool(new_company.strip())
        has_location = bool(new_location.strip())
        has_description = len(new_description.strip()) > 100
        
        # Need company + at least one other field for reliable dedup
        has_sufficient_data = has_company and (has_location or has_description)
        
        if has_sufficient_data:
            effective_threshold = self.similarity_threshold
        elif has_company:
            # Have company but no location/description - use stricter threshold
            effective_threshold = 0.98
            logger.debug(f"Using stricter threshold (0.98) - have company but missing location/description: {new_title[:50]}")
        else:
            # No company at all - essentially disable cross-source dedup (only exact matches)
            effective_threshold = 0.999
            logger.debug(f"Using very strict threshold (0.999) due to missing company for: {new_title[:50]}")
        
        best_match_id = None
        best_match_url = None
        best_match_signature = None
        best_similarity = 0.0
        
        for job_id, title, company, location, description, url in existing_jobs:
            # Check cache first
            if job_id in self._embeddings_cache:
                existing_embedding = self._embeddings_cache[job_id]
                existing_signature = self._signatures_cache[job_id]
            else:
                existing_signature = self.get_job_signature(title, company, location, description)
                existing_embedding = self.compute_embedding(existing_signature)
                self._embeddings_cache[job_id] = existing_embedding
                self._signatures_cache[job_id] = existing_signature
            
            # Cosine similarity (embeddings are already normalized)
            similarity = float(np.dot(new_embedding, existing_embedding))
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = job_id
                best_match_url = url
                best_match_signature = existing_signature
        
        is_duplicate = best_similarity >= effective_threshold
        
        # Critical: If locations differ, it's NOT a duplicate even if title/company match
        # Different locations means different jobs, even if semantically similar
        if is_duplicate and best_match_id is not None:
            # Get the matched job's location from existing_jobs
            for job_id, title, company, location, description, url in existing_jobs:
                if job_id == best_match_id:
                    # Normalize locations for comparison (strip extra whitespace)
                    new_loc_norm = new_location.strip().lower()
                    existing_loc_norm = location.strip().lower()
                    
                    # If locations are both present and differ, reject the duplicate
                    if new_loc_norm and existing_loc_norm and new_loc_norm != existing_loc_norm:
                        logger.info(
                            f"REJECTED DUPLICATE - Different locations (similarity={best_similarity:.3f}):\n"
                            f"  NEW ({new_location}): {new_url}\n"
                            f"  EXISTING (id={best_match_id}, {location}): {best_match_url}"
                        )
                        is_duplicate = False
                    break
        
        if is_duplicate:
            logger.info(
                f"DUPLICATE DETECTED (similarity={best_similarity:.3f}):\n"
                f"  NEW: {new_url}\n"
                f"       -> \"{new_signature[:100]}...\"\n"
                f"  EXISTING (id={best_match_id}): {best_match_url}\n"
                f"       -> \"{best_match_signature[:100]}...\""
            )
        elif best_similarity > 0.8:
            # Log near-misses for potential threshold tuning
            logger.debug(
                f"Near-duplicate (similarity={best_similarity:.3f}):\n"
                f"  NEW: {new_url}\n"
                f"       -> \"{new_signature[:100]}...\"\n"
                f"  CLOSEST (id={best_match_id}): {best_match_url}\n"
                f"       -> \"{best_match_signature[:100]}...\""
            )
        
        return DuplicateMatch(
            is_duplicate=is_duplicate,
            matched_job_id=best_match_id if is_duplicate else None,
            matched_url=best_match_url if is_duplicate else None,
            similarity_score=best_similarity,
            new_signature=new_signature,
            matched_signature=best_match_signature if is_duplicate else None
        )
    
    def batch_compute_embeddings(
        self,
        jobs: list[tuple[int, str, str, str, str]]  # (id, title, company, location, description)
    ) -> None:
        """
        Pre-compute and cache embeddings for a batch of jobs.
        
        Useful for warming up the cache with existing database jobs.
        
        Args:
            jobs: List of jobs as (id, title, company, location, description) tuples.
        """
        if not jobs:
            return
        
        self._ensure_model_loaded()
        model = _get_model()
        
        # Filter out already cached jobs
        uncached = [(j[0], j[1], j[2], j[3], j[4]) for j in jobs if j[0] not in self._embeddings_cache]
        
        if not uncached:
            logger.debug("All jobs already in embedding cache")
            return
        
        logger.info(f"Computing embeddings for {len(uncached)} jobs (batch)")
        
        # Create signatures
        signatures = []
        job_ids = []
        for job_id, title, company, location, description in uncached:
            sig = self.get_job_signature(title, company, location, description)
            signatures.append(sig)
            job_ids.append(job_id)
            self._signatures_cache[job_id] = sig
        
        # Batch encode (much faster than one-by-one)
        embeddings = model.encode(signatures, normalize_embeddings=True, show_progress_bar=False)
        
        # Cache results
        for job_id, embedding in zip(job_ids, embeddings):
            self._embeddings_cache[job_id] = embedding
        
        logger.info(f"Cached {len(uncached)} job embeddings")
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embeddings_cache.clear()
        self._signatures_cache.clear()
        logger.debug("Cleared embedding cache")
    
    def get_cache_stats(self) -> dict:
        """Get statistics about the embedding cache."""
        return {
            "cached_jobs": len(self._embeddings_cache),
            "model_loaded": self._model_loaded,
            "similarity_threshold": self.similarity_threshold
        }
