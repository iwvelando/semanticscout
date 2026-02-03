"""
Configuration management for Semantic Scout.

Handles loading and validating configuration from YAML files.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class LLMConfig:
    """Configuration for the local LLM via Ollama."""
    model: str = "llama3.1:8b"
    base_url: str = "http://localhost:11434"
    timeout: int = 120
    temperature: float = 0.3
    max_tokens: int = 500  # Output token limit (num_predict)
    context_window: int = 8192  # Total context window (num_ctx) for input + output
    # LLM service restart/retry configuration
    restart_url: Optional[str] = None  # URL for restarting the LLM service (e.g., http://localhost:8080/restart)
    restart_method: str = "POST"  # HTTP method for restart request
    restart_headers: Optional[dict[str, str]] = field(default_factory=lambda: {"Content-Type": "application/json"})
    restart_payload: Optional[dict] = field(default_factory=lambda: {"action": "restart", "mode": "replace"})
    restart_wait_seconds: int = 10  # Seconds to wait after restart before retry
    retry_attempts: int = 1  # Number of times to retry with restart on failure


@dataclass
class CityConfig:
    """Configuration for a single target city/location."""
    name: str  # The stanza name (e.g., "nyc", "chicago")
    target_latitude: float
    target_longitude: float
    radius_miles: float = 25.0


@dataclass
class LocationConfig:
    """Geographic targeting configuration.
    
    Supports multiple target cities. If cities is empty, geo filtering
    is disabled (all jobs pass), but include_remote is still honored.
    """
    include_remote: bool = True
    cities: dict[str, CityConfig] = field(default_factory=dict)
    # LLM-based geocoding fallback settings
    llm_geocode_fallback: bool = True  # Use LLM to resolve ambiguous locations
    llm_geocode_queries: int = 5  # Number of LLM queries for consensus
    llm_geocode_min_agreement: int = 3  # Minimum agreement for consensus (must be <= queries)
    
    @property
    def geo_filtering_enabled(self) -> bool:
        """Returns True if geographic filtering is enabled (at least one city defined)."""
        return len(self.cities) > 0


@dataclass
class SearchConfig:
    """Job search parameters."""
    keywords: list[str] = field(default_factory=lambda: [
        "Workday Manager",
        "HRIS Integration Architect",
        "Workday Studio Specialist",
        "Workday Integration Lead",
        "Workday PECI Developer",
        "HRIS Manager",
        "Workday Prism Analytics",
        "Workday BIRT Developer",
        "Workday Technical Lead",
    ])
    job_boards: list[str] = field(default_factory=lambda: [
        "linkedin",
        "indeed",
    ])
    max_results_per_keyword: int = 50
    job_goal_statement: str = (
        "Seeking a Workday leadership or senior IC role with deep technical focus on "
        "integrations, specifically Studio, PECI, BIRT, and Prism. Ideal position involves "
        "building complex integrations, managing hybrid IC/Manager teams, and leveraging "
        "HRIS expertise. Prefer roles emphasizing hands-on development alongside strategic "
        "oversight of Workday ecosystem implementations."
    )


@dataclass
class DatabaseConfig:
    """Database configuration."""
    db_path: str = "jobs.db"


@dataclass
class SlackConfig:
    """Slack notification configuration."""
    enabled: bool = False
    # Use either webhook_url OR bot_token + channel_id
    webhook_url: Optional[str] = None  # For simple message posting (no threading)
    bot_token: Optional[str] = None  # For threaded messages (requires chat:write scope)
    channel_id: Optional[str] = None  # Required when using bot_token
    # Message customization
    mention_users: list[str] = field(default_factory=list)  # User IDs to @mention
    jobs_per_thread: int = 5  # Number of jobs per threaded reply


@dataclass
class ReportingConfig:
    """Reporting configuration."""
    output_dir: str = "reports"
    min_score_threshold: float = 7.0
    include_job_descriptions: bool = True
    # Output options
    save_to_file: bool = True  # Save report to local file
    slack: SlackConfig = field(default_factory=SlackConfig)


@dataclass
class DeduplicationConfig:
    """Cross-source job deduplication configuration."""
    enabled: bool = True
    similarity_threshold: float = 0.92  # Cosine similarity threshold for duplicates
                                         # Higher = stricter (fewer false positives)
                                         # 0.92 is a good balance for job listings


@dataclass
class ScraperConfig:
    """Web scraper configuration."""
    headless: bool = True
    timeout: int = 30
    max_concurrent_requests: int = 5
    retry_attempts: int = 3
    retry_delay: float = 2.0
    parallel_boards: bool = True  # If True, search different job boards in parallel
    indeed_page_delay: float = 3.0  # Delay between Indeed pagination (helps avoid captchas)
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )


@dataclass
class Config:
    """Main configuration container."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    location: LocationConfig = field(default_factory=LocationConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    deduplication: DeduplicationConfig = field(default_factory=DeduplicationConfig)
    scraper: ScraperConfig = field(default_factory=ScraperConfig)

    def validate(self) -> list[str]:
        """
        Validate the configuration and return a list of errors.
        
        Returns:
            List of validation error messages. Empty if valid.
        """
        errors = []
        
        # Validate city configurations (if any defined)
        for city_name, city_config in self.location.cities.items():
            if not -90 <= city_config.target_latitude <= 90:
                errors.append(
                    f"City '{city_name}': target_latitude must be between -90 and 90, "
                    f"got {city_config.target_latitude}"
                )
            
            if not -180 <= city_config.target_longitude <= 180:
                errors.append(
                    f"City '{city_name}': target_longitude must be between -180 and 180, "
                    f"got {city_config.target_longitude}"
                )
            
            if city_config.radius_miles <= 0:
                errors.append(
                    f"City '{city_name}': radius_miles must be positive, "
                    f"got {city_config.radius_miles}"
                )
        
        # Note: Empty cities dict is valid (disables geo filtering, user open to anywhere)
        
        # Validate LLM configuration
        if not self.llm.model:
            errors.append("LLM model must be specified")
        
        # Validate search configuration
        if not self.search.keywords:
            errors.append("At least one search keyword must be specified")
        
        if not self.search.job_goal_statement:
            errors.append("job_goal_statement must be specified")
        
        return errors


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
                    Defaults to 'config.yaml' in the current directory.
    
    Returns:
        Validated Config object.
    
    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        ValueError: If the configuration is invalid.
    """
    if config_path is None:
        config_path = os.environ.get("SEMANTICSCOUT_CONFIG", "config.yaml")
    
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}. "
            "Please create a config.yaml file or set SEMANTICSCOUT_CONFIG environment variable."
        )
    
    with open(path, "r") as f:
        raw_config = yaml.safe_load(f) or {}
    
    config = Config()
    
    # Load LLM config
    if "llm" in raw_config:
        llm_data = raw_config["llm"]
        config.llm = LLMConfig(
            model=llm_data.get("model", config.llm.model),
            base_url=llm_data.get("base_url", config.llm.base_url),
            timeout=llm_data.get("timeout", config.llm.timeout),
            temperature=llm_data.get("temperature", config.llm.temperature),
            max_tokens=llm_data.get("max_tokens", config.llm.max_tokens),
            context_window=llm_data.get("context_window", config.llm.context_window),
            restart_url=llm_data.get("restart_url", config.llm.restart_url),
            restart_method=llm_data.get("restart_method", config.llm.restart_method),
            restart_headers=llm_data.get("restart_headers", config.llm.restart_headers),
            restart_payload=llm_data.get("restart_payload", config.llm.restart_payload),
            restart_wait_seconds=llm_data.get("restart_wait_seconds", config.llm.restart_wait_seconds),
            retry_attempts=llm_data.get("retry_attempts", config.llm.retry_attempts),
        )
    
    # Load location config
    if "location" in raw_config:
        loc_data = raw_config["location"]
        
        # Parse city stanzas - any key that's a dict with lat/long/radius is a city
        cities: dict[str, CityConfig] = {}
        reserved_keys = {
            "include_remote", "llm_geocode_fallback", 
            "llm_geocode_queries", "llm_geocode_min_agreement"
        }
        
        for key, value in loc_data.items():
            if key in reserved_keys:
                continue
            # Check if this is a city stanza (dict with target_latitude)
            if isinstance(value, dict) and "target_latitude" in value:
                cities[key] = CityConfig(
                    name=key,
                    target_latitude=value.get("target_latitude", 0.0),
                    target_longitude=value.get("target_longitude", 0.0),
                    radius_miles=value.get("radius_miles", 25.0),
                )
        
        config.location = LocationConfig(
            include_remote=loc_data.get("include_remote", config.location.include_remote),
            cities=cities,
            llm_geocode_fallback=loc_data.get("llm_geocode_fallback", config.location.llm_geocode_fallback),
            llm_geocode_queries=loc_data.get("llm_geocode_queries", config.location.llm_geocode_queries),
            llm_geocode_min_agreement=loc_data.get("llm_geocode_min_agreement", config.location.llm_geocode_min_agreement),
        )
    
    # Load search config
    if "search" in raw_config:
        search_data = raw_config["search"]
        config.search = SearchConfig(
            keywords=search_data.get("keywords", config.search.keywords),
            job_boards=search_data.get("job_boards", config.search.job_boards),
            max_results_per_keyword=search_data.get(
                "max_results_per_keyword", config.search.max_results_per_keyword
            ),
            job_goal_statement=search_data.get(
                "job_goal_statement", config.search.job_goal_statement
            ),
        )
    
    # Load database config
    if "database" in raw_config:
        db_data = raw_config["database"]
        config.database = DatabaseConfig(
            db_path=db_data.get("db_path", config.database.db_path),
        )
    
    # Load reporting config
    if "reporting" in raw_config:
        report_data = raw_config["reporting"]
        
        # Parse Slack config if present
        slack_config = SlackConfig()
        if "slack" in report_data:
            slack_data = report_data["slack"]
            slack_config = SlackConfig(
                enabled=slack_data.get("enabled", False),
                webhook_url=slack_data.get("webhook_url"),
                bot_token=slack_data.get("bot_token"),
                channel_id=slack_data.get("channel_id"),
                mention_users=slack_data.get("mention_users", []),
                jobs_per_thread=slack_data.get("jobs_per_thread", 5),
            )
        
        config.reporting = ReportingConfig(
            output_dir=report_data.get("output_dir", config.reporting.output_dir),
            min_score_threshold=report_data.get(
                "min_score_threshold", config.reporting.min_score_threshold
            ),
            include_job_descriptions=report_data.get(
                "include_job_descriptions", config.reporting.include_job_descriptions
            ),
            save_to_file=report_data.get("save_to_file", config.reporting.save_to_file),
            slack=slack_config,
        )
    
    # Load deduplication config
    if "deduplication" in raw_config:
        dedup_data = raw_config["deduplication"]
        config.deduplication = DeduplicationConfig(
            enabled=dedup_data.get("enabled", config.deduplication.enabled),
            similarity_threshold=dedup_data.get(
                "similarity_threshold", config.deduplication.similarity_threshold
            ),
        )
    
    # Load scraper config
    if "scraper" in raw_config:
        scraper_data = raw_config["scraper"]
        config.scraper = ScraperConfig(
            headless=scraper_data.get("headless", config.scraper.headless),
            timeout=scraper_data.get("timeout", config.scraper.timeout),
            max_concurrent_requests=scraper_data.get(
                "max_concurrent_requests", config.scraper.max_concurrent_requests
            ),
            retry_attempts=scraper_data.get("retry_attempts", config.scraper.retry_attempts),
            retry_delay=scraper_data.get("retry_delay", config.scraper.retry_delay),
            parallel_boards=scraper_data.get("parallel_boards", config.scraper.parallel_boards),
            indeed_page_delay=scraper_data.get("indeed_page_delay", config.scraper.indeed_page_delay),
            user_agent=scraper_data.get("user_agent", config.scraper.user_agent),
        )
    
    # Validate configuration
    errors = config.validate()
    if errors:
        raise ValueError(
            "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )
    
    return config


def generate_example_config(output_path: str = "config.example.yaml") -> None:
    """
    Generate an example configuration file with all available options.
    
    Args:
        output_path: Path where the example config will be written.
    """
    example_config = """# Semantic Scout Configuration
# Copy this file to config.yaml and customize for your needs.

# LLM Configuration (via Ollama)
llm:
  model: "llama3.1:8b"  # Model optimized for 8GB VRAM
  base_url: "http://localhost:11434"
  timeout: 120  # seconds
  temperature: 0.3
  max_tokens: 500

# Geographic Targeting
# Supports multiple cities. Jobs match if within ANY city's radius.
# Leave cities empty to disable geo filtering (open to anywhere).
location:
  include_remote: true  # Include remote positions
  
  # Define target cities (stanza names can be anything)
  nyc:
    target_latitude: 40.7128
    target_longitude: -74.0060
    radius_miles: 25.0
  
  # Add more cities if considering relocation:
  # chicago:
  #   target_latitude: 41.8781
  #   target_longitude: -87.6298
  #   radius_miles: 30.0
  
  llm_geocode_fallback: true
  llm_geocode_queries: 5
  llm_geocode_min_agreement: 3

# Job Search Parameters
search:
  keywords:
    - "Workday Manager"
    - "HRIS Integration Architect"
    - "Workday Studio Specialist"
    - "Workday Integration Lead"
    - "Workday PECI Developer"
    - "HRIS Manager"
    - "Workday Prism Analytics"
    - "Workday BIRT Developer"
    - "Workday Technical Lead"
  job_boards:
    - "linkedin"
    - "indeed"
  max_results_per_keyword: 50
  job_goal_statement: |
    Seeking a Workday leadership or senior IC role with deep technical focus on
    integrations, specifically Studio, PECI, BIRT, and Prism. Ideal position involves
    building complex integrations, managing hybrid IC/Manager teams, and leveraging
    HRIS expertise. Prefer roles emphasizing hands-on development alongside strategic
    oversight of Workday ecosystem implementations.

# Database Settings
database:
  db_path: "jobs.db"

# Reporting Settings
reporting:
  output_dir: "reports"
  min_score_threshold: 7.0  # Only include jobs scoring above this
  include_job_descriptions: true

# Web Scraper Settings
scraper:
  headless: true
  timeout: 30
  max_concurrent_requests: 5
  retry_attempts: 3
  retry_delay: 2.0
"""
    
    with open(output_path, "w") as f:
        f.write(example_config)
    
    print(f"Example configuration written to: {output_path}")
