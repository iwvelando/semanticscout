# Semantic Scout

**Local AI Job Intelligence Engine**

A Python-based automation pipeline that scrapes job listings, filters them by geographical radius, and uses a local LLM (via Ollama) to perform semantic scoring to identify high-value job opportunities.

## Features

- 🔍 **Multi-source Job Collection**: Scrapes jobs from LinkedIn, Indeed, and other job boards
- 🗺️ **Geographic Filtering**: Filter jobs by distance from your target location (with remote job detection)
- 🤖 **AI-Powered Scoring**: Uses local LLM via Ollama to score job relevance (1-10 scale)
- 💾 **Persistent Storage**: SQLite database for tracking jobs and preventing duplicates
- 📊 **Automated Reporting**: Daily markdown summaries of high-scoring opportunities
- 📈 **Application Tracking**: Track your application status for each job

## Requirements

- Python 3.11+
- Ubuntu 24.04 (Server) or similar Linux distribution
- NVIDIA GPU with 8GB+ VRAM (for local LLM inference)
- [Ollama](https://ollama.ai/) installed and running
- Playwright browsers installed

## Installation

### 1. Clone and Set Up Virtual Environment

```bash
cd /path/to/semanticscout
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Playwright Browsers

```bash
playwright install chromium
```

### 4. Install and Configure Ollama

```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the recommended model
ollama pull llama3.1:8b

# Start Ollama (if not running as a service)
ollama serve
```

### 5. Configure the Application

```bash
# Copy the example configuration
cp config.example.yaml config.yaml

# Edit with your settings
nano config.yaml
```

**Important**: You MUST configure the following in `config.yaml`:

- `location.target_latitude` and `location.target_longitude` - Your target location
- Optionally customize `search.keywords` and `search.job_goal_statement`

## Usage

### Run the Full Pipeline

```bash
python main.py
```

This will:
1. Scrape job listings from configured sources
2. Deduplicate and store new jobs
3. Apply geographic filtering
4. Score jobs using the local LLM
5. Generate a daily report

### Command Line Options

```bash
# Run with verbose logging
python main.py -v

# Use a different config file
python main.py -c /path/to/config.yaml

# Score only (skip job collection)
python main.py --score-only

# Generate reports only
python main.py --report daily
python main.py --report weekly
python main.py --report tracker

# Show database statistics
python main.py --stats

# Generate example config file
python main.py --init-config

# Log to file
python main.py --log-file semantic_scout.log
```

## Configuration

### LLM Settings

```yaml
llm:
  model: "llama3.1:8b"  # Optimized for 8GB VRAM
  base_url: "http://localhost:11434"
  timeout: 120
  temperature: 0.3
  max_tokens: 500
```

### Location Settings

```yaml
location:
  target_latitude: 40.7128  # Your latitude
  target_longitude: -74.0060  # Your longitude
  radius_miles: 10.0  # Search radius
  include_remote: true  # Include remote jobs
```

### Search Keywords

```yaml
search:
  keywords:
    - "Workday Manager"
    - "HRIS Integration Architect"
    - "Workday Studio Specialist"
  job_goal_statement: |
    Your detailed job goal statement here...
```

## Project Structure

```
semanticscout/
├── main.py                    # Main entry point
├── config.yaml               # Your configuration (create from example)
├── config.example.yaml       # Example configuration
├── requirements.txt          # Python dependencies
├── jobs.db                   # SQLite database (created on first run)
├── reports/                  # Generated reports
│   ├── daily_report_YYYY-MM-DD.md
│   └── weekly_summary_YYYY-MM-DD.md
└── semanticscout/            # Core modules
    ├── __init__.py
    ├── config.py             # Configuration management
    ├── database.py           # SQLite database layer
    ├── collector.py          # Job scraping (LinkedIn, Indeed)
    ├── geofence.py           # Geographic filtering
    ├── scorer.py             # LLM-based semantic scoring
    └── reporting.py          # Report generation
```

## Scoring System

Jobs are scored from 1-10 based on how well they match your job goal statement:

- **9-10**: Exceptional match - Perfect alignment with goals
- **7-8**: Strong match - Close alignment, likely a good fit
- **5-6**: Moderate match - Some alignment but gaps exist
- **3-4**: Weak match - Minimal alignment
- **1-2**: Poor match - Does not match goals

The daily report includes all jobs scoring **7 or above** (configurable).

## Application Status Tracking

Update job application status using the database directly or future CLI commands:

- `new` - Just discovered
- `reviewed` - You've looked at it
- `applied` - Application submitted
- `interviewing` - In interview process
- `offer` - Received an offer
- `rejected` - Application rejected
- `declined` - You declined
- `accepted` - Offer accepted

## Troubleshooting

### Ollama Connection Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve
```

### Model Not Found

```bash
# Pull the model
ollama pull llama3.1:8b

# List available models
ollama list
```

### Playwright Issues

```bash
# Install browsers
playwright install chromium

# Install system dependencies (Ubuntu)
playwright install-deps
```

### Geocoding Rate Limits

The geocoder uses Nominatim (OpenStreetMap) which has rate limits. If you encounter issues:
- Reduce the number of jobs per search
- Add delays between searches
- Consider using a paid geocoding service

## Development

### Running Tests

```bash
pip install pytest pytest-asyncio
pytest tests/
```

### Code Formatting

```bash
pip install black
black semanticscout/ main.py
```

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- [Ollama](https://ollama.ai/) for local LLM inference
- [Playwright](https://playwright.dev/) for web scraping
- [geopy](https://geopy.readthedocs.io/) for geocoding
