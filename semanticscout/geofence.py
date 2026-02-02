"""
Geofence module for Semantic Scout.

Provides geographic filtering of job listings based on distance from target location.
Uses geocoding to convert location strings to coordinates and Haversine formula
for distance calculation. Includes LLM-based fallback for ambiguous locations.
"""

import asyncio
import logging
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import httpx
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

from .config import LocationConfig, LLMConfig
from .database import Job

logger = logging.getLogger(__name__)


# Common remote work indicators
REMOTE_INDICATORS = [
    r'\bremote\b',
    r'\bwork from home\b',
    r'\bwfh\b',
    r'\btelecommute\b',
    r'\btelecommuting\b',
    r'\bvirtual\b',
    r'\bdistributed\b',
    r'\banywhere\b',
    r'\bhybrid remote\b',
    r'\bfully remote\b',
    r'\b100% remote\b',
    r'\bwork remotely\b',
    r'\bremote work\b',
    r'\bremote position\b',
    r'\bremote opportunity\b',
    r'\bremote-friendly\b',
    r'\blocation flexible\b',
    r'\bflexible location\b',
]

# Country-level locations that typically indicate remote (no city/state specified)
# These are checked separately as they need exact matching
COUNTRY_ONLY_LOCATIONS = [
    "united states",
    "usa",
    "us",
    "united states of america",
    "canada",
    "united kingdom",
    "uk",
    "worldwide",
    "global",
    "north america",
    "europe",
    "emea",
    "apac",
    "americas",
]

# Compile regex patterns
REMOTE_PATTERN = re.compile('|'.join(REMOTE_INDICATORS), re.IGNORECASE)


@dataclass
class GeocodedLocation:
    """Result of geocoding a location string."""
    original: str
    latitude: Optional[float]
    longitude: Optional[float]
    is_remote: bool
    formatted_address: Optional[str] = None
    geocoding_failed: bool = False


@dataclass
class GeoFilterResult:
    """Result of geographic filtering for a job."""
    passes: bool
    is_remote: bool
    distance_miles: Optional[float]
    matched_city: Optional[str]  # Name of city stanza that matched, or "Remote"


class Geofencer:
    """
    Geographic filter for job listings.
    
    Supports multiple target cities. If no cities are configured, geo filtering
    is disabled (all jobs pass based on remote preference). Uses LLM fallback
    for ambiguous locations.
    """
    
    def __init__(self, config: LocationConfig, llm_config: Optional[LLMConfig] = None):
        """
        Initialize the geofencer.
        
        Args:
            config: Location configuration with target cities and settings.
            llm_config: Optional LLM configuration for fallback geocoding.
        """
        self.config = config
        self.llm_config = llm_config
        self.cities = config.cities  # Dict of city_name -> CityConfig
        self.include_remote = config.include_remote
        self.geo_filtering_enabled = config.geo_filtering_enabled
        
        # LLM fallback settings
        self.llm_geocode_fallback = config.llm_geocode_fallback
        self.llm_geocode_queries = config.llm_geocode_queries
        self.llm_geocode_min_agreement = config.llm_geocode_min_agreement
        
        # Initialize geocoder with a reasonable timeout
        self.geocoder = Nominatim(
            user_agent="semanticscout/1.0",
            timeout=10
        )
        
        # Cache for geocoded locations to reduce API calls
        self._geocode_cache: dict[str, GeocodedLocation] = {}
        # Cache for LLM location resolutions
        self._llm_location_cache: dict[str, Optional[str]] = {}
        
        # Log configuration
        if self.geo_filtering_enabled:
            city_info = ", ".join(
                f"{name}=({c.target_latitude:.2f}, {c.target_longitude:.2f}, {c.radius_miles}mi)"
                for name, c in self.cities.items()
            )
            logger.info(
                f"Geofencer initialized with {len(self.cities)} cities: {city_info}, "
                f"include_remote={self.include_remote}, llm_fallback={self.llm_geocode_fallback}"
            )
        else:
            logger.info(
                f"Geofencer initialized: geo filtering DISABLED (no cities defined), "
                f"include_remote={self.include_remote}"
            )
    
    def is_remote(self, location_str: str, description: str = "") -> bool:
        """
        Check if a job is remote based on location string and description.
        
        Args:
            location_str: The job's location string.
            description: The job description text.
        
        Returns:
            True if the job appears to be remote.
        """
        combined_text = f"{location_str} {description}"
        
        # Check regex patterns first
        if REMOTE_PATTERN.search(combined_text):
            return True
        
        # Check if location is just a country/region (indicates remote)
        location_normalized = location_str.lower().strip()
        # Remove common punctuation and extra whitespace
        location_normalized = ' '.join(location_normalized.split())
        
        for country_location in COUNTRY_ONLY_LOCATIONS:
            if location_normalized == country_location:
                logger.debug(f"Location '{location_str}' matches country-only pattern, treating as remote")
                return True
        
        return False
    
    async def geocode(self, location_str: str) -> GeocodedLocation:
        """
        Geocode a location string to coordinates.
        
        Handles semicolon-delimited multi-location strings by trying each
        location individually and returning the first successful geocode.
        
        Args:
            location_str: Location string (e.g., "San Francisco, CA" or 
                         "Austin, TX, USA; Dallas, TX, USA").
        
        Returns:
            GeocodedLocation with coordinates if successful.
        """
        # Check cache first
        cache_key = location_str.lower().strip()
        if cache_key in self._geocode_cache:
            return self._geocode_cache[cache_key]
        
        # Check if it's a remote job
        is_remote = self.is_remote(location_str)
        
        # Handle multi-location strings (semicolon-separated)
        locations = [loc.strip() for loc in location_str.split(';') if loc.strip()]
        if not locations:
            locations = [location_str]
        
        # Try to geocode - for multi-location, try each until one succeeds
        result = GeocodedLocation(
            original=location_str,
            latitude=None,
            longitude=None,
            is_remote=is_remote,
        )
        
        for location in locations:
            single_result = await self._geocode_single_location(location)
            
            # If this location is remote, mark the result as remote
            if single_result.is_remote:
                result.is_remote = True
            
            # If we got coordinates, use them and stop
            if single_result.latitude is not None and single_result.longitude is not None:
                result.latitude = single_result.latitude
                result.longitude = single_result.longitude
                result.formatted_address = single_result.formatted_address
                if len(locations) > 1:
                    logger.debug(f"Multi-location geocode: used '{location}' from '{location_str}'")
                break
        
        # If no location geocoded successfully, mark as failed
        if result.latitude is None and result.longitude is None and not result.is_remote:
            result.geocoding_failed = True
            if len(locations) > 1:
                logger.debug(f"Could not geocode any of {len(locations)} locations in: {location_str}")
        
        self._geocode_cache[cache_key] = result
        return result
    
    async def _geocode_single_location(self, location_str: str) -> GeocodedLocation:
        """
        Geocode a single location string (no semicolons).
        
        Args:
            location_str: A single location string.
        
        Returns:
            GeocodedLocation with coordinates if successful.
        """
        # Check cache first
        cache_key = location_str.lower().strip()
        if cache_key in self._geocode_cache:
            return self._geocode_cache[cache_key]
        
        # Check if it's a remote indicator
        is_remote = self.is_remote(location_str)
        
        result = GeocodedLocation(
            original=location_str,
            latitude=None,
            longitude=None,
            is_remote=is_remote,
        )
        
        # Clean up location string for geocoding
        clean_location = self._clean_location_string(location_str)
        
        if not clean_location or clean_location.lower() in ["remote", "anywhere", "virtual"]:
            result.is_remote = True
            self._geocode_cache[cache_key] = result
            return result
        
        # Try geocoding the cleaned location
        geo_result = self._try_geocode(clean_location)
        
        if geo_result:
            result.latitude = geo_result.latitude
            result.longitude = geo_result.longitude
            result.formatted_address = geo_result.address
            logger.debug(f"Geocoded '{location_str}' -> ({result.latitude}, {result.longitude})")
        else:
            # If initial geocoding failed, try LLM fallback
            if self.llm_geocode_fallback and self.llm_config:
                llm_resolved = await self._resolve_location_with_llm(location_str)
                if llm_resolved:
                    logger.info(f"LLM resolved '{location_str}' to '{llm_resolved}'")
                    geo_result = self._try_geocode(llm_resolved)
                    if geo_result:
                        result.latitude = geo_result.latitude
                        result.longitude = geo_result.longitude
                        result.formatted_address = geo_result.address
                        logger.debug(f"Geocoded LLM-resolved '{llm_resolved}' -> ({result.latitude}, {result.longitude})")
                    else:
                        logger.debug(f"Could not geocode LLM-resolved location: {llm_resolved}")
                        result.geocoding_failed = True
                else:
                    logger.debug(f"LLM could not resolve location: {location_str}")
                    result.geocoding_failed = True
            else:
                logger.debug(f"Could not geocode location: {location_str}")
                result.geocoding_failed = True
        
        self._geocode_cache[cache_key] = result
        return result
    
    def _try_geocode(self, location_str: str):
        """
        Attempt to geocode a location string.
        
        Args:
            location_str: Cleaned location string.
        
        Returns:
            Geocoding result or None if failed.
        """
        try:
            return self.geocoder.geocode(location_str, exactly_one=True)
        except GeocoderTimedOut:
            logger.warning(f"Geocoding timeout for: {location_str}")
            return None
        except GeocoderServiceError as e:
            logger.error(f"Geocoding service error for '{location_str}': {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected geocoding error for '{location_str}': {e}")
            return None
    
    async def _resolve_location_with_llm(self, location_str: str) -> Optional[str]:
        """
        Use LLM to resolve an ambiguous location to a City, State format.
        
        Uses consensus voting: queries the LLM multiple times and only
        accepts a result if there's sufficient agreement.
        
        Includes automatic retry with Ollama restart on failures.
        
        Args:
            location_str: The ambiguous location string.
        
        Returns:
            Resolved "City, State" string or None if no consensus.
        """
        # Check LLM cache first
        cache_key = location_str.lower().strip()
        if cache_key in self._llm_location_cache:
            return self._llm_location_cache[cache_key]
        
        if not self.llm_config:
            return None
        
        # Retry logic
        max_attempts = 1 + (self.llm_config.retry_attempts if self.llm_config else 0)
        
        for attempt in range(max_attempts):
            try:
                result = await self._query_llm_for_location(location_str)
                
                # Cache and return result (even if None)
                self._llm_location_cache[cache_key] = result
                return result
                
            except Exception as e:
                logger.error(f"LLM location resolution error (attempt {attempt + 1}/{max_attempts}): {e}")
                
                # If this was the last attempt, cache None and return
                if attempt == max_attempts - 1:
                    self._llm_location_cache[cache_key] = None
                    return None
                
                # Restart Ollama and retry
                if self.llm_config and self.llm_config.restart_url:
                    logger.info(f"Attempting Ollama restart before retry {attempt + 2}/{max_attempts}")
                    await self._restart_ollama()
                else:
                    # Brief delay before retry even without restart
                    await asyncio.sleep(2)
        
        self._llm_location_cache[cache_key] = None
        return None
    
    async def _restart_ollama(self) -> bool:
        """
        Restart Ollama service by POSTing to the configured restart URL.
        
        Returns:
            True if restart request succeeded, False otherwise.
        """
        if not self.llm_config or not self.llm_config.restart_url:
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
    
    async def _query_llm_for_location(self, location_str: str) -> Optional[str]:
        """
        Query LLM multiple times to resolve location with consensus.
        
        Args:
            location_str: The ambiguous location string.
        
        Returns:
            Resolved "City, State" string or None if no consensus.
            
        Raises:
            Exception: If LLM queries fail (e.g., timeout, 502 error).
        """
        prompt = (
            f'What is the best City, State assignment for "{location_str}"? '
            'Your answer must be strictly in the form of "City, State" and should not '
            'include any other text UNLESS you cannot make an educated best guess in '
            'which case your answer is to be "Unknown"'
        )
        
        responses = []
        last_error = None
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            for i in range(self.llm_geocode_queries):
                try:
                    response = await client.post(
                        f"{self.llm_config.base_url}/api/generate",
                        json={
                            "model": self.llm_config.model,
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "temperature": 0.7,  # Higher temp for diversity
                                "num_predict": 50,
                            }
                        }
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        raw_response = result.get("response", "").strip()
                        
                        # Clean and validate the response
                        cleaned = self._parse_llm_location_response(raw_response)
                        if cleaned:
                            responses.append(cleaned)
                            logger.debug(f"LLM query {i+1}: '{location_str}' -> '{cleaned}'")
                    else:
                        # Track API errors (like 502) to potentially trigger restart
                        last_error = Exception(f"Ollama API error: {response.status_code}")
                        logger.debug(f"LLM query {i+1} failed: {response.status_code}")
                        
                except httpx.TimeoutException as e:
                    last_error = e
                    logger.debug(f"LLM query {i+1} timed out")
                except Exception as e:
                    last_error = e
                    logger.debug(f"LLM query {i+1} failed: {e}")
        
        # If we got no valid responses and had errors, raise to trigger retry
        if not responses and last_error:
            raise last_error
        
        if not responses:
            logger.warning(f"No valid LLM responses for location: {location_str}")
            return None
        
        # Find consensus
        counter = Counter(responses)
        most_common, count = counter.most_common(1)[0]
        
        if count >= self.llm_geocode_min_agreement:
            logger.info(
                f"LLM consensus ({count}/{len(responses)}): "
                f"'{location_str}' -> '{most_common}'"
            )
            return most_common
        else:
            logger.warning(
                f"No LLM consensus for '{location_str}': "
                f"responses={dict(counter)}, needed {self.llm_geocode_min_agreement}"
            )
            return None
    
    def _parse_llm_location_response(self, response: str) -> Optional[str]:
        """
        Parse and validate LLM location response.
        
        Args:
            response: Raw LLM response.
        
        Returns:
            Cleaned "City, State" string or None if invalid.
        """
        if not response:
            return None
        
        # Clean up the response
        cleaned = response.strip().strip('"\'')
        
        # Check for "Unknown" response
        if cleaned.lower() in ["unknown", "n/a", "none", "not found"]:
            return None
        
        # Validate it looks like "City, State" format
        # Should have a comma and be reasonably short
        if ',' not in cleaned:
            return None
        
        if len(cleaned) > 100:  # Too long, probably got extra text
            return None
        
        # Extract just the first line if multi-line
        cleaned = cleaned.split('\n')[0].strip()
        
        # Remove any trailing periods or extra punctuation
        cleaned = cleaned.rstrip('.')
        
        # Basic format check: should have 2 parts separated by comma
        parts = [p.strip() for p in cleaned.split(',')]
        if len(parts) < 2:
            return None
        
        # Take first two parts (City, State)
        city_state = f"{parts[0]}, {parts[1]}"
        
        return city_state
    
    def _clean_location_string(self, location_str: str) -> str:
        """
        Clean a location string for better geocoding results.
        
        Args:
            location_str: Raw location string.
        
        Returns:
            Cleaned location string.
        """
        # Remove common noise
        cleaned = location_str.strip()
        
        # Remove "Hybrid work in", "On-site in", "Remote in", etc.
        work_location_prefixes = [
            r'^hybrid\s+(?:work\s+)?in\s+',
            r'^on-?site\s+(?:work\s+)?in\s+',
            r'^remote\s+(?:work\s+)?in\s+',
            r'^in-?office\s+(?:work\s+)?in\s+',
            r'^work\s+(?:from\s+)?in\s+',
            r'^located\s+in\s+',
            r'^based\s+in\s+',
            r'^position\s+in\s+',
            r'^job\s+in\s+',
        ]
        for prefix_pattern in work_location_prefixes:
            cleaned = re.sub(prefix_pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Remove remote indicators that might confuse geocoding
        for pattern in REMOTE_INDICATORS:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Remove parenthetical content
        cleaned = re.sub(r'\([^)]*\)', '', cleaned)
        
        # Remove common suffixes like "Metropolitan Area", "Metroplex", "Greater ... Area"
        metro_suffixes = [
            r'\s+metropolitan\s+area$',
            r'\s+metro\s+area$',
            r'\s+metroplex$',
            r'\s+area$',
            r'^greater\s+',
        ]
        for suffix_pattern in metro_suffixes:
            cleaned = re.sub(suffix_pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Remove extra whitespace
        cleaned = ' '.join(cleaned.split())
        
        # Remove trailing punctuation
        cleaned = cleaned.rstrip(',.-')
        
        return cleaned.strip()
    
    def calculate_distance_from_city(
        self, 
        city_name: str,
        latitude: float, 
        longitude: float
    ) -> float:
        """
        Calculate distance in miles from a specific target city.
        
        Args:
            city_name: Name of the city stanza.
            latitude: Job location latitude.
            longitude: Job location longitude.
        
        Returns:
            Distance in miles.
        """
        city = self.cities[city_name]
        target = (city.target_latitude, city.target_longitude)
        job_location = (latitude, longitude)
        distance = geodesic(target, job_location).miles
        return round(distance, 2)
    
    def find_matching_city(
        self, 
        latitude: float, 
        longitude: float
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        Find the first city whose radius contains the given coordinates.
        
        Args:
            latitude: Job location latitude.
            longitude: Job location longitude.
        
        Returns:
            Tuple of (city_name, distance_miles) if within any city's radius,
            otherwise (None, None).
        """
        job_location = (latitude, longitude)
        
        for city_name, city_config in self.cities.items():
            target = (city_config.target_latitude, city_config.target_longitude)
            distance = geodesic(target, job_location).miles
            
            if distance <= city_config.radius_miles:
                return city_name, round(distance, 2)
        
        return None, None
    
    async def filter_job(self, job: Job) -> GeoFilterResult:
        """
        Determine if a job passes the geographic filter.
        
        Args:
            job: The job to filter.
        
        Returns:
            GeoFilterResult with pass/fail status and match details.
        """
        # First check if it's remote
        is_remote = self.is_remote(job.location, job.description)
        
        # Handle remote jobs
        if is_remote:
            if self.include_remote:
                logger.debug(f"Job '{job.title}' at '{job.location}' passes as remote")
                return GeoFilterResult(
                    passes=True,
                    is_remote=True,
                    distance_miles=None,
                    matched_city="Remote"
                )
            else:
                # Remote not allowed
                logger.debug(f"Job '{job.title}' is remote but include_remote=False")
                return GeoFilterResult(
                    passes=False,
                    is_remote=True,
                    distance_miles=None,
                    matched_city=None
                )
        
        # If no cities configured, geo filtering is disabled
        # Non-remote jobs pass only if include_remote is False (user wants non-remote only)
        # OR pass always if user is open to relocating anywhere (empty cities)
        if not self.geo_filtering_enabled:
            # No cities = open to anywhere, all non-remote jobs pass
            logger.debug(
                f"Job '{job.title}' at '{job.location}' passes (geo filtering disabled)"
            )
            return GeoFilterResult(
                passes=True,
                is_remote=False,
                distance_miles=None,
                matched_city="Anywhere"  # Indicates no geo filter applied
            )
        
        # Handle multi-location jobs (semicolon-separated)
        # Split and try each location - if ANY is within geofence, job passes
        locations = [loc.strip() for loc in job.location.split(';') if loc.strip()]
        
        # If no valid locations parsed, treat as single location
        if not locations:
            locations = [job.location]
        
        # Track if we're dealing with multi-location and results
        is_multi_location = len(locations) > 1
        if is_multi_location:
            logger.debug(f"Job '{job.title}' has {len(locations)} locations, checking each")
        
        # Try each location
        for location in locations:
            result = await self._check_single_location(job, location)
            if result.passes:
                if is_multi_location:
                    logger.debug(
                        f"Job '{job.title}' matches via location '{location}' "
                        f"(city={result.matched_city}, distance={result.distance_miles}mi)"
                    )
                return result
        
        # None of the locations matched
        if is_multi_location:
            logger.debug(f"Job '{job.title}' - none of {len(locations)} locations matched geofence")
        
        # Return failure with distance to nearest city from first location attempt
        # We need to return a meaningful distance for logging
        return await self._get_failure_result(job, locations[0])
    
    async def _check_single_location(self, job: Job, location: str) -> GeoFilterResult:
        """
        Check if a single location string passes the geofence filter.
        
        Args:
            job: The job being filtered.
            location: A single location string to check.
        
        Returns:
            GeoFilterResult for this location.
        """
        # Check if this specific location indicates remote
        if self.is_remote(location, ""):
            if self.include_remote:
                return GeoFilterResult(
                    passes=True,
                    is_remote=True,
                    distance_miles=None,
                    matched_city="Remote"
                )
            else:
                return GeoFilterResult(
                    passes=False,
                    is_remote=True,
                    distance_miles=None,
                    matched_city=None
                )
        
        # Get coordinates - check if already cached on job (only for primary location)
        latitude = None
        longitude = None
        
        # Try to geocode this location
        geocoded = await self.geocode(location)
        
        if geocoded.is_remote:
            if self.include_remote:
                return GeoFilterResult(
                    passes=True,
                    is_remote=True,
                    distance_miles=None,
                    matched_city="Remote"
                )
            else:
                return GeoFilterResult(
                    passes=False,
                    is_remote=True,
                    distance_miles=None,
                    matched_city=None
                )
        
        if geocoded.latitude is None or geocoded.longitude is None:
            # Could not geocode this location - continue to next
            logger.debug(f"Could not geocode location '{location}' for job '{job.title}'")
            return GeoFilterResult(
                passes=False,
                is_remote=False,
                distance_miles=None,
                matched_city=None
            )
        
        latitude = geocoded.latitude
        longitude = geocoded.longitude
        
        # Check against all cities
        matched_city, distance = self.find_matching_city(latitude, longitude)
        
        if matched_city:
            return GeoFilterResult(
                passes=True,
                is_remote=False,
                distance_miles=distance,
                matched_city=matched_city
            )
        
        # This location didn't match
        return GeoFilterResult(
            passes=False,
            is_remote=False,
            distance_miles=distance,  # Will be None since we didn't calculate it here
            matched_city=None
        )
    
    async def _get_failure_result(self, job: Job, primary_location: str) -> GeoFilterResult:
        """
        Generate a failure result with distance info for logging.
        
        Args:
            job: The job that failed filtering.
            primary_location: The primary location to use for distance calculation.
        
        Returns:
            GeoFilterResult indicating failure.
        """
        # Try to get coordinates for logging distance
        geocoded = await self.geocode(primary_location)
        
        if geocoded.latitude and geocoded.longitude and self.cities:
            min_city = None
            min_distance = float('inf')
            for city_name in self.cities:
                dist = self.calculate_distance_from_city(
                    city_name, geocoded.latitude, geocoded.longitude
                )
                if dist < min_distance:
                    min_distance = dist
                    min_city = city_name
            
            logger.debug(
                f"Job '{job.title}' at '{job.location}' is {min_distance:.1f} miles "
                f"from nearest city '{min_city}' (outside radius)"
            )
            return GeoFilterResult(
                passes=False,
                is_remote=False,
                distance_miles=min_distance,
                matched_city=None
            )
        
        return GeoFilterResult(
            passes=False,
            is_remote=False,
            distance_miles=None,
            matched_city=None
        )
    
    async def enrich_job_location(self, job: Job) -> Job:
        """
        Enrich a job with geocoded coordinates.
        
        Note: Does NOT set distance_miles or matched_city - that happens
        during filter_job() when we know which city matched.
        
        Args:
            job: The job to enrich.
        
        Returns:
            Job with updated location data.
        """
        # Check for remote first
        is_remote = self.is_remote(job.location, job.description)
        job.is_remote = is_remote
        
        # Get coordinates if not already present
        if job.latitude is None or job.longitude is None:
            geocoded = await self.geocode(job.location)
            job.latitude = geocoded.latitude
            job.longitude = geocoded.longitude
            job.is_remote = geocoded.is_remote or is_remote
        
        return job
    
    async def filter_jobs(self, jobs: list[Job]) -> list[Job]:
        """
        Filter a list of jobs by geographic criteria.
        
        Args:
            jobs: List of jobs to filter.
        
        Returns:
            List of jobs that pass the geographic filter.
        """
        passed = []
        
        for job in jobs:
            result = await self.filter_job(job)
            
            if result.passes:
                # Enrich with location data
                job.is_remote = result.is_remote
                job.distance_miles = result.distance_miles
                job.matched_city = result.matched_city
                passed.append(job)
        
        logger.info(f"Geographic filter: {len(passed)}/{len(jobs)} jobs passed")
        return passed
    
    def clear_cache(self) -> None:
        """Clear the geocoding cache."""
        self._geocode_cache.clear()
        logger.debug("Geocoding cache cleared")
