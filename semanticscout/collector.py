"""
Job collector module for Semantic Scout.

Implements async web scraping for job listings from LinkedIn, Indeed, and company boards
using Playwright for dynamic content rendering.
"""

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from urllib.parse import quote_plus, urljoin

from playwright.async_api import Browser, Page, async_playwright, TimeoutError as PlaywrightTimeout

from .config import ScraperConfig
from .database import Job

logger = logging.getLogger(__name__)


@dataclass
class RawJobListing:
    """Raw job listing data before processing."""
    title: str
    company: str
    location: str
    description: str
    url: str
    source: str
    posted_date: Optional[str] = None
    salary: Optional[str] = None
    job_type: Optional[str] = None
    raw_data: Optional[dict] = None


class JobCollector(ABC):
    """Abstract base class for job collectors."""
    
    @abstractmethod
    async def search(self, keyword: str, max_results: int = 50) -> list[RawJobListing]:
        """Search for jobs with the given keyword.
        
        Args:
            keyword: Search keyword.
            max_results: Maximum number of results to fetch.
        
        Note: Initial collection is always fast (no full descriptions).
              Full descriptions are fetched separately for geo-filtered jobs.
        """
        pass
    
    @abstractmethod
    async def fetch_job_description(self, url: str) -> Optional[str]:
        """Fetch the full job description from a job URL.
        
        Args:
            url: The job listing URL.
        
        Returns:
            Full job description text, or None if failed.
        """
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        pass


class LinkedInCollector(JobCollector):
    """Collector for LinkedIn job listings."""
    
    BASE_URL = "https://www.linkedin.com/jobs/search"
    
    def __init__(self, browser: Browser, config: ScraperConfig):
        self.browser = browser
        self.config = config
        self.source = "linkedin"
    
    async def search(self, keyword: str, max_results: int = 50) -> list[RawJobListing]:
        """
        Search LinkedIn for jobs matching the keyword.
        
        Performs fast collection (no full descriptions) for efficiency.
        Full descriptions are fetched separately for geo-filtered jobs.
        
        Args:
            keyword: Search keyword.
            max_results: Maximum number of results to fetch.
        
        Returns:
            List of raw job listings (with snippet descriptions only).
        """
        jobs = []
        context = await self.browser.new_context(
            user_agent=self.config.user_agent,
            viewport={"width": 1920, "height": 1080}
        )
        
        try:
            page = await context.new_page()
            
            # Build search URL
            search_url = f"{self.BASE_URL}?keywords={quote_plus(keyword)}&location=United%20States"
            logger.info(f"LinkedIn search URL: {search_url}")
            
            await page.goto(search_url, timeout=self.config.timeout * 1000)
            await page.wait_for_load_state("networkidle", timeout=self.config.timeout * 1000)
            
            # Wait for job listings to load
            try:
                await page.wait_for_selector(".jobs-search__results-list", timeout=10000)
            except PlaywrightTimeout:
                logger.warning("LinkedIn job list not found, page structure may have changed")
                return jobs
            
            # Scroll to load more jobs
            scroll_count = min(max_results // 25, 4)  # LinkedIn loads ~25 at a time
            for i in range(scroll_count):
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await asyncio.sleep(0.5)  # Reduced from 1s
            
            # Extract job cards
            job_cards = await page.query_selector_all(".jobs-search__results-list > li")
            total_cards = len(job_cards)
            cards_to_process = job_cards[:max_results]
            logger.info(f"Found {total_cards} job cards on LinkedIn, processing up to {len(cards_to_process)} (max_results={max_results})")
            
            for idx, card in enumerate(cards_to_process):
                try:
                    job = await self._extract_job_from_card(card)
                    if job:
                        jobs.append(job)
                except Exception as e:
                    logger.error(f"Error extracting LinkedIn job: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"LinkedIn search error: {e}")
        finally:
            await context.close()
        
        return jobs
    
    async def _extract_job_from_card(self, card) -> Optional[RawJobListing]:
        """Extract job data from a LinkedIn job card (fast mode - no clicking)."""
        try:
            # Get basic info from card
            title_elem = await card.query_selector(".base-search-card__title")
            company_elem = await card.query_selector(".base-search-card__subtitle")
            location_elem = await card.query_selector(".job-search-card__location")
            link_elem = await card.query_selector("a.base-card__full-link")
            
            if not all([title_elem, company_elem, location_elem, link_elem]):
                return None
            
            title = await title_elem.inner_text()
            company = await company_elem.inner_text()
            location = await location_elem.inner_text()
            url = await link_elem.get_attribute("href")
            
            # Get description snippet from card (fast - no clicking)
            description = ""
            snippet_elem = await card.query_selector(".base-search-card__metadata, .job-search-card__snippet")
            if snippet_elem:
                description = await snippet_elem.inner_text()
            
            return RawJobListing(
                title=title.strip(),
                company=company.strip(),
                location=location.strip(),
                description=description.strip(),
                url=url.strip() if url else "",
                source=self.source,
            )
        except Exception as e:
            logger.error(f"Error parsing LinkedIn card: {e}")
            return None
    
    async def fetch_job_description(self, url: str) -> Optional[str]:
        """
        Fetch the full job description from a LinkedIn job URL.
        
        Args:
            url: The LinkedIn job listing URL.
        
        Returns:
            Full job description text, or None if failed.
        """
        context = await self.browser.new_context(
            user_agent=self.config.user_agent,
            viewport={"width": 1920, "height": 1080}
        )
        
        try:
            page = await context.new_page()
            await page.goto(url, timeout=self.config.timeout * 1000)
            await page.wait_for_load_state("domcontentloaded", timeout=self.config.timeout * 1000)
            
            # Wait for description to load
            await asyncio.sleep(1)
            
            # Try to find description element
            desc_elem = await page.query_selector(".show-more-less-html__markup")
            if desc_elem:
                description = await desc_elem.inner_text()
                return description.strip()
            
            # Try alternative selectors
            alt_selectors = [
                ".description__text",
                ".jobs-description-content__text",
                "[data-job-description]",
            ]
            
            for selector in alt_selectors:
                desc_elem = await page.query_selector(selector)
                if desc_elem:
                    description = await desc_elem.inner_text()
                    return description.strip()
            
            logger.warning(f"Could not find description on LinkedIn page: {url}")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching LinkedIn description from {url}: {e}")
            return None
        finally:
            await context.close()
    
    async def close(self) -> None:
        """Clean up resources."""
        pass


class IndeedCollector(JobCollector):
    """Collector for Indeed job listings."""
    
    BASE_URL = "https://www.indeed.com/jobs"
    
    def __init__(self, browser: Browser, config: ScraperConfig):
        self.browser = browser
        self.config = config
        self.source = "indeed"
    
    async def search(self, keyword: str, max_results: int = 50) -> list[RawJobListing]:
        """
        Search Indeed for jobs matching the keyword.
        
        Performs fast collection (no full descriptions) for efficiency.
        Full descriptions are fetched separately for geo-filtered jobs.
        
        Args:
            keyword: Search keyword.
            max_results: Maximum number of results to fetch.
        
        Returns:
            List of raw job listings (with snippet descriptions only).
        """
        jobs = []
        context = await self.browser.new_context(
            user_agent=self.config.user_agent,
            viewport={"width": 1920, "height": 1080}
        )
        
        try:
            page = await context.new_page()
            
            # Paginate through results
            start = 0
            pages_without_results = 0
            
            while len(jobs) < max_results and pages_without_results < 2:
                search_url = f"{self.BASE_URL}?q={quote_plus(keyword)}&start={start}"
                logger.info(f"Indeed search URL: {search_url}")
                
                try:
                    # Navigate with extended timeout, don't wait for networkidle (can hang)
                    await page.goto(search_url, timeout=60000, wait_until="domcontentloaded")
                    
                    # Wait a bit for dynamic content
                    await asyncio.sleep(2)
                    
                    # Try multiple selectors for job listings
                    job_cards = []
                    
                    # Try different possible selectors
                    selectors_to_try = [
                        ".job_seen_beacon",
                        ".jobsearch-ResultsList .result",
                        "[data-testid='jobListing']",
                        ".tapItem",
                        ".resultContent",
                        ".jobsearch-SerpJobCard",
                        "div.slider_container .slider_item",
                    ]
                    
                    for selector in selectors_to_try:
                        try:
                            await page.wait_for_selector(selector, timeout=3000)
                            job_cards = await page.query_selector_all(selector)
                            if job_cards:
                                logger.debug(f"Indeed: Found {len(job_cards)} jobs using selector '{selector}'")
                                break
                        except PlaywrightTimeout:
                            continue
                    
                    if not job_cards:
                        # Check if we hit a captcha or block page
                        page_content = await page.content()
                        if "captcha" in page_content.lower() or "unusual traffic" in page_content.lower():
                            logger.warning("Indeed may be blocking requests (captcha detected)")
                            break
                        
                        logger.warning(f"No job cards found on Indeed page (start={start})")
                        pages_without_results += 1
                        start += 10
                        continue
                    
                    pages_without_results = 0
                    jobs_on_page = len(job_cards)
                    logger.info(f"Found {jobs_on_page} job cards on Indeed page")
                    
                    # Extract all job data using JavaScript to avoid stale element issues
                    extracted_jobs = await self._extract_jobs_from_page(page)
                    
                    for job_data in extracted_jobs:
                        if len(jobs) >= max_results:
                            break
                        
                        if job_data:
                            jobs.append(job_data)
                    
                    logger.info(f"Extracted {len(extracted_jobs)} jobs from Indeed page")
                    
                    # Move to next page
                    start += 10
                    if start >= max_results:
                        break
                    
                    # Longer delay between pages to avoid captchas
                    page_delay = getattr(self.config, 'indeed_page_delay', 3.0)
                    await asyncio.sleep(page_delay)
                    
                except PlaywrightTimeout as e:
                    logger.warning(f"Indeed page timeout (start={start}): {e}")
                    pages_without_results += 1
                    start += 10
                    continue
                except Exception as e:
                    logger.error(f"Indeed page error: {e}")
                    break
            
        except Exception as e:
            logger.error(f"Indeed search error: {e}")
        finally:
            await context.close()
        
        return jobs
    
    async def _extract_jobs_from_page(self, page: Page) -> list[RawJobListing]:
        """
        Extract all job data from the current Indeed page using JavaScript.
        
        This approach is more robust than querying individual elements,
        as it extracts all data in a single evaluation before the DOM can change.
        """
        try:
            # Use JavaScript to extract all job data at once
            jobs_data = await page.evaluate("""
                () => {
                    const jobs = [];
                    
                    // Try multiple selectors for job cards
                    const cardSelectors = [
                        '.job_seen_beacon',
                        '.jobsearch-ResultsList .result',
                        '.tapItem',
                        '.resultContent'
                    ];
                    
                    let cards = [];
                    for (const selector of cardSelectors) {
                        cards = document.querySelectorAll(selector);
                        if (cards.length > 0) break;
                    }
                    
                    cards.forEach(card => {
                        try {
                            // Extract title
                            let title = '';
                            const titleSelectors = [
                                'h2.jobTitle span[title]',
                                'h2.jobTitle a',
                                '[data-testid="jobTitle"]',
                                '.jobTitle a',
                                '.jobTitle span',
                                'a.jcs-JobTitle'
                            ];
                            for (const sel of titleSelectors) {
                                const el = card.querySelector(sel);
                                if (el) {
                                    title = el.getAttribute('title') || el.textContent || '';
                                    if (title.trim()) break;
                                }
                            }
                            
                            // Extract company
                            let company = 'Unknown Company';
                            const companySelectors = [
                                '[data-testid="company-name"]',
                                '.companyName',
                                '.company',
                                'span.companyName'
                            ];
                            for (const sel of companySelectors) {
                                const el = card.querySelector(sel);
                                if (el && el.textContent.trim()) {
                                    company = el.textContent.trim();
                                    break;
                                }
                            }
                            
                            // Extract location
                            let location = '';
                            const locationSelectors = [
                                '[data-testid="text-location"]',
                                '.companyLocation',
                                '.location'
                            ];
                            for (const sel of locationSelectors) {
                                const el = card.querySelector(sel);
                                if (el && el.textContent.trim()) {
                                    location = el.textContent.trim();
                                    break;
                                }
                            }
                            
                            // Extract URL
                            let url = '';
                            const linkSelectors = ['h2.jobTitle a', 'a.jcs-JobTitle', 'a[data-jk]', '.jobTitle a'];
                            for (const sel of linkSelectors) {
                                const el = card.querySelector(sel);
                                if (el) {
                                    const href = el.getAttribute('href');
                                    if (href) {
                                        url = href.startsWith('http') ? href : 'https://www.indeed.com' + href;
                                        break;
                                    }
                                }
                            }
                            
                            // Extract description snippet
                            let description = '';
                            const descSelectors = ['.job-snippet', '[data-testid="jobDescriptionText"]', '.summary'];
                            for (const sel of descSelectors) {
                                const el = card.querySelector(sel);
                                if (el && el.textContent.trim()) {
                                    description = el.textContent.trim();
                                    break;
                                }
                            }
                            
                            if (title.trim()) {
                                jobs.push({
                                    title: title.trim(),
                                    company: company,
                                    location: location,
                                    description: description,
                                    url: url
                                });
                            }
                        } catch (e) {
                            // Skip this card if there's an error
                        }
                    });
                    
                    return jobs;
                }
            """)
            
            # Convert to RawJobListing objects
            result = []
            for job_data in jobs_data:
                result.append(RawJobListing(
                    title=job_data.get('title', ''),
                    company=job_data.get('company', 'Unknown Company'),
                    location=job_data.get('location', ''),
                    description=job_data.get('description', ''),
                    url=job_data.get('url', ''),
                    source=self.source,
                ))
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting Indeed jobs via JS: {e}")
            return []
    
    async def _extract_job_from_card(
        self, 
        page: Page, 
        card,
        fast_mode: bool = False
    ) -> Optional[RawJobListing]:
        """Extract job data from an Indeed job card (legacy method, kept for reference)."""
        try:
            # Get title - try multiple selectors
            title = None
            title_selectors = [
                'h2.jobTitle span[title]',
                'h2.jobTitle a',
                '[data-testid="jobTitle"]',
                '.jobTitle',
                'a.jcs-JobTitle',
            ]
            
            for selector in title_selectors:
                title_elem = await card.query_selector(selector)
                if title_elem:
                    title = await title_elem.get_attribute("title")
                    if not title:
                        title = await title_elem.inner_text()
                    if title:
                        break
            
            if not title:
                return None
            
            # Get company - try multiple selectors
            company = "Unknown Company"
            company_selectors = [
                '[data-testid="company-name"]',
                '.companyName',
                '.company',
                'span.companyName',
            ]
            
            for selector in company_selectors:
                company_elem = await card.query_selector(selector)
                if company_elem:
                    company = await company_elem.inner_text()
                    if company:
                        break
            
            # Get location - try multiple selectors
            location = ""
            location_selectors = [
                '[data-testid="text-location"]',
                '.companyLocation',
                '.location',
            ]
            
            for selector in location_selectors:
                location_elem = await card.query_selector(selector)
                if location_elem:
                    location = await location_elem.inner_text()
                    if location:
                        break
            
            # Get URL - prefer data-jk attribute for reliable job key
            url = ""
            job_key = None
            
            # First try to get job key from data-jk attribute
            jk_elem = await card.query_selector('a[data-jk]')
            if jk_elem:
                job_key = await jk_elem.get_attribute('data-jk')
            
            if job_key:
                # Use canonical viewjob URL
                url = f"https://www.indeed.com/viewjob?jk={job_key}"
            else:
                # Fallback to href and extract jk parameter
                link_selectors = ['h2.jobTitle a', 'a.jcs-JobTitle', 'a[data-jk]']
                
                for selector in link_selectors:
                    link_elem = await card.query_selector(selector)
                    if link_elem:
                        href = await link_elem.get_attribute("href")
                        if href:
                            # Extract job key from redirect URL
                            jk_match = re.search(r'jk=([a-f0-9]+)', href)
                            if jk_match:
                                job_key = jk_match.group(1)
                                url = f"https://www.indeed.com/viewjob?jk={job_key}"
                            else:
                                url = urljoin("https://www.indeed.com", href)
                            break
            
            # Get description snippet
            description = ""
            desc_selectors = [
                '.job-snippet',
                '[data-testid="jobDescriptionText"]',
                '.summary',
            ]
            
            for selector in desc_selectors:
                desc_elem = await card.query_selector(selector)
                if desc_elem:
                    description = await desc_elem.inner_text()
                    if description:
                        break
            
            return RawJobListing(
                title=title.strip(),
                company=company.strip(),
                location=location.strip(),
                description=description.strip(),
                url=url,
                source=self.source,
            )
        except Exception as e:
            logger.error(f"Error parsing Indeed card: {e}")
            return None
    
    async def fetch_job_description(self, url: str) -> Optional[str]:
        """
        Fetch the full job description from an Indeed job URL.
        
        Indeed aggressively blocks headless browsers, causing timeouts.
        We skip full description fetching and rely on snippets captured
        during the initial search results scrape.
        
        Args:
            url: The Indeed job listing URL.
        
        Returns:
            None - Indeed description fetching is disabled.
        """
        # Indeed blocks headless browsers aggressively - skip full fetch
        # The snippet captured during search is used for scoring instead
        logger.debug(f"Skipping Indeed full description fetch (blocked by site): {url}")
        return None
    
    async def close(self) -> None:
        """Clean up resources."""
        pass


class DiceCollector(JobCollector):
    """Collector for Dice job listings (tech-focused job board)."""
    
    BASE_URL = "https://www.dice.com/jobs"
    
    def __init__(self, browser: Browser, config: ScraperConfig):
        self.browser = browser
        self.config = config
        self.source = "dice"
    
    async def search(self, keyword: str, max_results: int = 50) -> list[RawJobListing]:
        """
        Search Dice for jobs matching the keyword.
        
        Args:
            keyword: Search keyword.
            max_results: Maximum number of results to fetch.
        
        Returns:
            List of raw job listings.
        """
        jobs = []
        context = await self.browser.new_context(
            user_agent=self.config.user_agent,
            viewport={"width": 1920, "height": 1080}
        )
        
        try:
            page = await context.new_page()
            
            # Build search URL - Dice updated their URL structure
            search_url = f"https://www.dice.com/jobs?q={quote_plus(keyword)}&location=United%20States&countryCode=US&page=1&pageSize=50"
            logger.info(f"Dice search URL: {search_url}")
            
            await page.goto(search_url, timeout=self.config.timeout * 1000)
            await page.wait_for_load_state("networkidle", timeout=self.config.timeout * 1000)
            
            # Give React/Vue app time to render
            await asyncio.sleep(3)
            
            # Dice uses a variety of selectors - try multiple
            # From HTML inspection: div[role="listitem"] > ... > div[data-testid="job-card"]
            card_selectors = [
                'div[data-testid="job-card"]',  # Primary: actual card container
                'div[role="article"]',  # The job-card also has role="article"
                'div[role="listitem"]',  # Parent container that wraps the full card
                'dhi-search-card',  # Web component (older Dice)
                '[data-testid="job-search-result"]',
                '[data-cy="search-result-job-card"]',
            ]
            
            job_cards = []
            used_selector = None
            for selector in card_selectors:
                job_cards = await page.query_selector_all(selector)
                if job_cards:
                    used_selector = selector
                    logger.debug(f"Dice: Found {len(job_cards)} cards with selector: {selector}")
                    break
            
            # If no cards found with proper selectors, try finding job links and navigate to parent
            if not job_cards:
                logger.debug("Dice: No cards found with standard selectors, trying link-based fallback")
                job_links = await page.query_selector_all('a[data-testid="job-search-job-detail-link"]')
                if not job_links:
                    job_links = await page.query_selector_all('a[href*="/job-detail/"]')
                
                if job_links:
                    logger.debug(f"Dice: Found {len(job_links)} job links, navigating to parent cards")
                    # Navigate each link to its parent card (div[data-testid="job-card"] or similar)
                    seen_cards = set()
                    for link in job_links:
                        try:
                            # Navigate up to find the card container
                            parent_card = await link.evaluate_handle("""
                                el => {
                                    let parent = el;
                                    while (parent && parent.parentElement) {
                                        parent = parent.parentElement;
                                        if (parent.getAttribute && 
                                            (parent.getAttribute('data-testid') === 'job-card' ||
                                             parent.getAttribute('role') === 'article' ||
                                             parent.getAttribute('data-job-guid'))) {
                                            return parent;
                                        }
                                    }
                                    return null;
                                }
                            """)
                            if parent_card:
                                # Use data-job-guid as unique identifier to avoid duplicates
                                card_id = await parent_card.evaluate("el => el.getAttribute('data-job-guid') || el.getAttribute('data-id') || ''")
                                if card_id and card_id not in seen_cards:
                                    seen_cards.add(card_id)
                                    job_cards.append(parent_card)
                        except Exception as e:
                            logger.debug(f"Dice: Error navigating to parent card: {e}")
                    
                    if job_cards:
                        used_selector = "link->parent navigation"
                        logger.debug(f"Dice: Found {len(job_cards)} unique parent cards via navigation")
            
            if not job_cards:
                # Debug: log what we see on the page
                title = await page.title()
                logger.warning(f"Dice job list not found. Page title: {title}")
                
                # Check if there's a "no results" message
                no_results = await page.query_selector('[class*="no-results"], [class*="NoResults"]')
                if no_results:
                    logger.info("Dice returned no results for this search")
                    return jobs
                
                # Log page HTML snippet for debugging (first 1000 chars of body)
                body = await page.query_selector('body')
                if body:
                    html = await body.inner_html()
                    logger.debug(f"Dice page body preview: {html[:1000]}...")
                
                return jobs
            
            total_cards = len(job_cards)
            cards_to_process = job_cards[:max_results]
            logger.info(f"Found {total_cards} job cards on Dice, processing up to {len(cards_to_process)}")
            
            for card in cards_to_process:
                try:
                    job = await self._extract_job_from_card(card, page)
                    if job:
                        jobs.append(job)
                except Exception as e:
                    logger.error(f"Error extracting Dice job: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Dice search error: {e}")
        finally:
            await context.close()
        
        return jobs
    
    async def _extract_job_from_card(self, card, page: Page) -> Optional[RawJobListing]:
        """Extract job data from a Dice job card."""
        try:
            # From HTML inspection, the title link has class "text-xl font-semibold text-zinc-800"
            # There are 3 links with /job-detail/: invisible overlay, Easy Apply button, and the actual title
            # We need to target the title specifically by its styling class
            title_selectors = [
                # Primary: title link has text-xl class (from HTML inspection)
                'a.text-xl[href*="/job-detail/"]',
                # Alternative: inside div.content, the title link
                'div.content a[href*="/job-detail/"]',
                # Legacy selectors
                'a[data-cy="card-title-link"]',
                'h5 a[href*="/job-detail/"]',
                'h4 a[href*="/job-detail/"]', 
                'h3 a[href*="/job-detail/"]',
            ]
            
            title_elem = None
            for sel in title_selectors:
                title_elem = await card.query_selector(sel)
                if title_elem:
                    # Verify this is actually the title (has text, not opacity-0)
                    text = await title_elem.inner_text()
                    if text.strip() and text.strip().lower() not in ['easy apply', 'apply now', '']:
                        logger.debug(f"Dice: Found title with selector: {sel}")
                        break
                    title_elem = None  # Reset if this wasn't a valid title
            
            if not title_elem:
                # Last resort: find all job-detail links and pick the one with actual job title text
                all_links = await card.query_selector_all('a[href*="/job-detail/"]')
                for link in all_links:
                    text = await link.inner_text()
                    text = text.strip()
                    # Skip empty, "Easy Apply", "Apply Now", or very short text
                    if text and len(text) > 5 and text.lower() not in ['easy apply', 'apply now', 'save', 'share']:
                        title_elem = link
                        logger.debug(f"Dice: Found title via link iteration: '{text[:30]}...'")
                        break
                
            if not title_elem:
                logger.debug("Dice: Could not find valid title link in card")
                return None
            
            # Get title text
            title = await title_elem.inner_text()
            url = await title_elem.get_attribute("href")
            
            # Filter out "Easy Apply" and other non-job-title text that gets captured
            # Common patterns: "Easy Apply", "Featured", "New", "Apply Now"
            invalid_titles = [
                "easy apply", "featured", "new", "hot", "promoted",
                "apply now", "apply", "save", "save job", "share",
                "view job", "see job", "learn more"
            ]
            title_clean = title.strip()
            
            # Strip badge patterns from start/end
            for pattern in invalid_titles:
                if title_clean.lower().startswith(pattern):
                    title_clean = title_clean[len(pattern):].strip()
                if title_clean.lower().endswith(pattern):
                    title_clean = title_clean[:-len(pattern)].strip()
            
            # If the entire title was just invalid text, skip
            if not title_clean or title_clean.lower() in invalid_titles:
                logger.debug(f"Dice: Skipping card with invalid title text: '{title}'")
                return None
            
            # Additional validation: real job titles are usually > 5 chars
            if len(title_clean) < 5:
                logger.debug(f"Dice: Skipping card with too-short title: '{title_clean}'")
                return None
            
            title = title_clean
            
            # Company - Dice structure from HTML inspection:
            # Find all p tags with line-clamp classes inside company-profile links
            # Then pick the one that's not inside an image div (the text one)
            company = ""
            try:
                # Get all company-profile links
                company_links = await card.query_selector_all('a[href*="/company-profile/"]')
                logger.debug(f"Dice: Found {len(company_links)} company-profile links")
                
                for i, link in enumerate(company_links):
                    # Skip the first link if it only contains an image (logo)
                    link_text = await link.inner_text()
                    link_text = link_text.strip()
                    logger.debug(f"Dice: Company link {i} text: '{link_text[:30]}'")
                    
                    # Look for p tags in this link
                    p_elems = await link.query_selector_all('p')
                    for p in p_elems:
                        p_text = await p.inner_text()
                        p_text = p_text.strip()
                        # Company name should have actual text, not be empty
                        if p_text and len(p_text) > 3:
                            company = p_text
                            logger.debug(f"Dice: Found company name: {company[:50]}")
                            break
                    
                    if company:
                        break
            except Exception as e:
                logger.debug(f"Dice: Error extracting company: {e}")
            
            # Location - Dice structure from HTML inspection:
            # Look for p tags with text-zinc-600 styling that contain location info
            # Pattern: "Remote or Boston, Massachusetts" or just "Remote"
            location = ""
            try:
                # Get all p elements with location styling
                p_elems = await card.query_selector_all('p.text-zinc-600')
                logger.debug(f"Dice: Found {len(p_elems)} p.text-zinc-600 elements")
                
                for p in p_elems:
                    text = await p.inner_text()
                    text = text.strip()
                    
                    # Location text should have comma (City, State) or contain "Remote"
                    # Should NOT be just a bullet point or time
                    if text and (',' in text or 'remote' in text.lower()) and '•' not in text:
                        # Filter out times like "Today" or date-like text
                        if not text.isdigit() and len(text) > 2:
                            location = text
                            logger.debug(f"Dice: Found location: {location}")
                            break
            except Exception as e:
                logger.debug(f"Dice: Error extracting location: {e}")
            
            # Description snippet
            description_selectors = [
                '[data-cy="card-summary"]',
                '[data-cy="card-description"]',
                '[class*="job-description"]',
                '[class*="snippet"]',
                '[class*="Summary"]',
            ]
            description = ""
            for sel in description_selectors:
                elem = await card.query_selector(sel)
                if elem:
                    description = await elem.inner_text()
                    if description.strip():
                        break
            
            # Ensure URL is absolute
            if url and not url.startswith("http"):
                url = f"https://www.dice.com{url}"
            
            if not title or not url:
                logger.debug(f"Dice: Skipping card with missing title/url. title='{title}', url='{url}'")
                return None
            
            # Log extracted data - use INFO level so it's visible by default
            logger.info(f"Dice extracted: title='{title[:40]}', company='{company}', location='{location}'")
            
            return RawJobListing(
                title=title.strip(),
                company=company.strip(),
                location=location.strip(),
                description=description.strip(),
                url=url.strip(),
                source=self.source,
            )
        except Exception as e:
            logger.error(f"Error parsing Dice card: {e}")
            return None
    
    async def fetch_job_description(self, url: str) -> Optional[str]:
        """
        Fetch the full job description from a Dice job URL.
        
        Args:
            url: The Dice job listing URL.
        
        Returns:
            Full job description text, or None if failed.
        """
        context = await self.browser.new_context(
            user_agent=self.config.user_agent,
            viewport={"width": 1920, "height": 1080}
        )
        
        try:
            page = await context.new_page()
            await page.goto(url, timeout=self.config.timeout * 1000)
            await page.wait_for_load_state("networkidle", timeout=self.config.timeout * 1000)
            
            # Wait for JS to render
            await asyncio.sleep(3)
            
            # Try to find description element - Dice uses various selectors
            desc_selectors = [
                '[data-testid="jobDescriptionHtml"]',
                '[data-cy="job-description"]',
                '#jobDescriptionHtml',
                '#jobdescSec',
                '.job-description',
                '[class*="jobDescription"]',
                '[class*="JobDescription"]',
                'div[class*="description"]',
                'section[class*="description"]',
            ]
            
            for selector in desc_selectors:
                desc_elem = await page.query_selector(selector)
                if desc_elem:
                    description = await desc_elem.inner_text()
                    if description and len(description.strip()) > 50:
                        return description.strip()
            
            logger.warning(f"Could not find description on Dice page: {url}")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching Dice description from {url}: {e}")
            return None
        finally:
            await context.close()
    
    async def close(self) -> None:
        """Clean up resources."""
        pass


class BuiltInCollector(JobCollector):
    """Collector for Built In job listings (tech/startup-focused job board)."""
    
    BASE_URL = "https://builtin.com/jobs"
    
    def __init__(self, browser: Browser, config: ScraperConfig):
        self.browser = browser
        self.config = config
        self.source = "builtin"
    
    async def search(self, keyword: str, max_results: int = 50) -> list[RawJobListing]:
        """
        Search Built In for jobs matching the keyword.
        
        Built In is a regional site - it covers major tech hubs.
        This collector searches the national view.
        
        Args:
            keyword: Search keyword.
            max_results: Maximum number of results to fetch.
        
        Returns:
            List of raw job listings.
        """
        jobs = []
        context = await self.browser.new_context(
            user_agent=self.config.user_agent,
            viewport={"width": 1920, "height": 1080}
        )
        
        try:
            page = await context.new_page()
            
            # Built In uses 'search' parameter
            search_url = f"{self.BASE_URL}?search={quote_plus(keyword)}"
            logger.info(f"Built In search URL: {search_url}")
            
            await page.goto(search_url, timeout=self.config.timeout * 1000)
            await page.wait_for_load_state("networkidle", timeout=self.config.timeout * 1000)
            
            # Wait for job listings to load
            try:
                await page.wait_for_selector('[data-id="job-card"], .job-card', timeout=15000)
            except PlaywrightTimeout:
                logger.warning("Built In job list not found, page structure may have changed")
                return jobs
            
            # Scroll to load more jobs
            scroll_count = min(max_results // 20, 3)
            for _ in range(scroll_count):
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await asyncio.sleep(1)
            
            # Extract job cards
            job_cards = await page.query_selector_all('[data-id="job-card"], .job-card, article[class*="job"]')
            total_cards = len(job_cards)
            cards_to_process = job_cards[:max_results]
            logger.info(f"Found {total_cards} job cards on Built In, processing up to {len(cards_to_process)}")
            
            for card in cards_to_process:
                try:
                    job = await self._extract_job_from_card(card, page)
                    if job:
                        jobs.append(job)
                except Exception as e:
                    logger.error(f"Error extracting Built In job: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Built In search error: {e}")
        finally:
            await context.close()
        
        return jobs
    
    async def _extract_job_from_card(self, card, page: Page) -> Optional[RawJobListing]:
        """Extract job data from a Built In job card."""
        try:
            # Debug: dump card HTML structure (first 500 chars)
            card_html = await card.inner_html()
            logger.debug(f"Built In card HTML preview: {card_html[:500]}...")
            
            # Get basic info from card - try multiple selector strategies
            title_selectors = [
                'h2 a', 'h3 a', 'a[data-id="job-title"]', '.job-title a',
                '[class*="JobTitle"] a', '[class*="job-title"] a', 'a[href*="/job/"]'
            ]
            title_elem = None
            for sel in title_selectors:
                title_elem = await card.query_selector(sel)
                if title_elem:
                    logger.debug(f"Built In: Found title with selector: {sel}")
                    break
            
            company_selectors = [
                '[data-id="company-title"]', '.company-name', 'a[href*="/company/"]',
                '[class*="CompanyName"]', '[class*="company-name"]', 'span[class*="company"]',
                'div[class*="Company"] a', 'a[class*="company"]'
            ]
            company_elem = None
            for sel in company_selectors:
                company_elem = await card.query_selector(sel)
                if company_elem:
                    logger.debug(f"Built In: Found company with selector: {sel}")
                    break
            
            location_selectors = [
                # Primary: span with font-barlow and text-gray-04 classes (from inspection)
                'span.font-barlow.text-gray-04',
                # Fallback selectors
                '[data-id="job-location"]', '.job-location',
            ]
            location_elem = None
            for sel in location_selectors:
                location_elem = await card.query_selector(sel)
                if location_elem:
                    logger.debug(f"Built In: Found location with selector: {sel}")
                    break
            
            if not title_elem:
                logger.warning("Built In: No title element found in card")
                return None
            
            title = await title_elem.inner_text()
            url = await title_elem.get_attribute("href")
            
            company = ""
            if company_elem:
                company = await company_elem.inner_text()
            else:
                logger.debug(f"Built In: No company element found for job: {title[:50]}")
            
            location = ""
            # The selector `span.font-barlow.text-gray-04` matches BOTH work arrangement (Hybrid, In-Office)
            # and actual location. We need to get all matching spans and filter to actual locations.
            all_spans = await card.query_selector_all('span.font-barlow.text-gray-04')
            for span in all_spans:
                text = await span.inner_text()
                text = text.strip()
                
                # Check for multi-location tooltip (data-bs-title attribute)
                # This will have text like "2 Locations" or "4 Locations"
                if text and text.lower().endswith('locations') and ' ' in text:
                    try:
                        tooltip_locations = await span.get_attribute("data-bs-title")
                        if tooltip_locations:
                            # Parse HTML like "<div class='text-truncate'>City, ST, USA</div>..."
                            cities = re.findall(r">([^<]+)</div>", tooltip_locations)
                            if cities:
                                location = "; ".join(cities)
                                logger.debug(f"Built In: Extracted {len(cities)} locations from tooltip: {location[:60]}...")
                                break
                    except Exception as e:
                        logger.debug(f"Built In: Error parsing multi-location tooltip: {e}")
                
                # Location patterns: "City, ST, Country", "Remote", etc.
                # Skip: "Hybrid", "In-Office", "On-site", "Remote or Hybrid"
                elif text and text.lower() not in ['hybrid', 'in-office', 'on-site', 'remote', 'remote or hybrid']:
                    # Check if it looks like a location (contains comma, or has country code)
                    if ',' in text or 'remote' in text.lower() or any(x in text for x in ['USA', 'UK', 'CA']):
                        location = text
                        logger.debug(f"Built In: Extracted location from span: {text}")
                        break
            
            if not location:
                logger.debug(f"Built In: No location found for job: {title[:50]}")
            
            # Get description snippet if available
            description = ""
            snippet_elem = await card.query_selector('[data-id="job-description"], .job-description, p')
            if snippet_elem:
                description = await snippet_elem.inner_text()
            
            # Log what we extracted
            logger.info(f"Built In extracted: title='{title[:40]}', company='{company}', location='{location}'")
            
            # Ensure URL is absolute
            if url and not url.startswith("http"):
                url = f"https://builtin.com{url}"
            
            return RawJobListing(
                title=title.strip(),
                company=company.strip() if company else "",
                location=location.strip() if location else "",
                description=description.strip(),
                url=url.strip() if url else "",
                source=self.source,
            )
        except Exception as e:
            logger.error(f"Error parsing Built In card: {e}")
            return None
    
    async def fetch_job_description(self, url: str) -> Optional[str]:
        """
        Fetch the full job description from a Built In job URL.
        
        Args:
            url: The Built In job listing URL.
        
        Returns:
            Full job description text, or None if failed.
        """
        context = await self.browser.new_context(
            user_agent=self.config.user_agent,
            viewport={"width": 1920, "height": 1080}
        )
        
        try:
            page = await context.new_page()
            await page.goto(url, timeout=self.config.timeout * 1000)
            await page.wait_for_load_state("domcontentloaded", timeout=self.config.timeout * 1000)
            
            # Wait for description to load
            await asyncio.sleep(2)
            
            # Try to find description element - Built In selectors
            # Built In has various page layouts depending on company
            desc_selectors = [
                '[data-id="job-description"]',
                '[data-testid="job-description"]',
                '.job-description',
                '.job-info__description',
                '[class*="JobDescription"]',
                '[class*="jobDescription"]',
                'article [class*="description"]',
                # Common content containers
                '.job-posting-description',
                '.posting-description',
                '#job-description',
                'section.description',
                # Fallback: main content area
                'main article',
                '.job-details-content',
                '[class*="JobDetails"] [class*="content"]',
            ]
            
            for selector in desc_selectors:
                desc_elem = await page.query_selector(selector)
                if desc_elem:
                    description = await desc_elem.inner_text()
                    if description and len(description.strip()) > 50:
                        return description.strip()
            
            # Last resort: try to find any large text block in the main area
            main_elem = await page.query_selector('main, article, .content, #content')
            if main_elem:
                paragraphs = await main_elem.query_selector_all('p')
                if len(paragraphs) > 2:
                    texts = []
                    for p in paragraphs[:20]:  # Limit to avoid too much text
                        text = await p.inner_text()
                        if text.strip():
                            texts.append(text.strip())
                    if texts:
                        return "\n\n".join(texts)
            
            logger.warning(f"Could not find description on Built In page: {url}")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching Built In description from {url}: {e}")
            return None
        finally:
            await context.close()
    
    async def close(self) -> None:
        """Clean up resources."""
        pass


class CollectorManager:
    """Manages multiple job collectors and coordinates searches."""
    
    def __init__(self, config: ScraperConfig):
        self.config = config
        self.browser = None
        self.playwright = None
        self.collectors: dict[str, JobCollector] = {}
    
    async def initialize(self) -> None:
        """Initialize the browser and collectors."""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=self.config.headless,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-infobars',
                '--no-sandbox',
            ]
        )
        
        # Initialize collectors
        self.collectors["linkedin"] = LinkedInCollector(self.browser, self.config)
        self.collectors["indeed"] = IndeedCollector(self.browser, self.config)
        self.collectors["dice"] = DiceCollector(self.browser, self.config)
        self.collectors["builtin"] = BuiltInCollector(self.browser, self.config)
        
        logger.info("Collector manager initialized with collectors: %s", 
                   list(self.collectors.keys()))
    
    async def _search_single_board(
        self,
        board: str,
        keywords: list[str],
        max_results_per_keyword: int,
    ) -> list[RawJobListing]:
        """
        Search a single job board for all keywords.
        
        Args:
            board: The job board to search.
            keywords: List of search keywords.
            max_results_per_keyword: Maximum results per keyword.
        
        Returns:
            List of all job listings from this board.
        """
        jobs = []
        
        if board not in self.collectors:
            logger.warning(f"Unknown job board: {board}")
            return jobs
        
        collector = self.collectors[board]
        
        for keyword in keywords:
            for attempt in range(self.config.retry_attempts):
                try:
                    logger.info(f"Searching {board} for '{keyword}' (attempt {attempt + 1})")
                    keyword_jobs = await collector.search(
                        keyword, 
                        max_results_per_keyword,
                    )
                    logger.info(f"Found {len(keyword_jobs)} jobs from {board} for '{keyword}'")
                    jobs.extend(keyword_jobs)
                    break
                except Exception as e:
                    logger.error(f"Search failed: {e}")
                    if attempt < self.config.retry_attempts - 1:
                        await asyncio.sleep(self.config.retry_delay)
                    else:
                        logger.error(f"All retry attempts failed for {board}/{keyword}")
            
            # Rate limiting between keyword searches on same board
            await asyncio.sleep(1)
        
        return jobs
    
    async def search_all(
        self, 
        keywords: list[str], 
        job_boards: list[str],
        max_results_per_keyword: int = 50,
        parallel: bool = True
    ) -> list[RawJobListing]:
        """
        Search all configured job boards for the given keywords.
        
        Performs fast collection (no full descriptions) for all jobs.
        Full descriptions should be fetched separately for geo-filtered jobs.
        
        Args:
            keywords: List of search keywords.
            job_boards: List of job boards to search.
            max_results_per_keyword: Maximum results per keyword per board.
            parallel: If True, search different job boards in parallel.
        
        Returns:
            Combined list of all job listings (with snippet descriptions only).
        """
        if parallel and len(job_boards) > 1:
            # Search different job boards in parallel
            logger.info(f"Searching {len(job_boards)} job boards in parallel: {job_boards}")
            
            tasks = [
                self._search_single_board(board, keywords, max_results_per_keyword)
                for board in job_boards
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            all_jobs = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error searching {job_boards[i]}: {result}")
                else:
                    all_jobs.extend(result)
            
            return all_jobs
        else:
            # Sequential search
            all_jobs = []
            for board in job_boards:
                board_jobs = await self._search_single_board(
                    board, keywords, max_results_per_keyword
                )
                all_jobs.extend(board_jobs)
            return all_jobs
    
    async def fetch_descriptions(
        self,
        jobs: list[Job],
        max_concurrent: int = 3
    ) -> dict[int, str]:
        """
        Fetch full job descriptions for a list of jobs.
        
        This is used after geo-filtering to get full context for scoring.
        
        Args:
            jobs: List of Job objects to fetch descriptions for.
            max_concurrent: Maximum concurrent fetches per source.
        
        Returns:
            Dictionary mapping job.id to full description text.
        """
        results = {}
        
        # Group jobs by source
        jobs_by_source: dict[str, list[Job]] = {}
        for job in jobs:
            if job.source not in jobs_by_source:
                jobs_by_source[job.source] = []
            jobs_by_source[job.source].append(job)
        
        logger.info(
            f"Fetching full descriptions for {len(jobs)} jobs "
            f"({', '.join(f'{k}: {len(v)}' for k, v in jobs_by_source.items())})"
        )
        
        for source, source_jobs in jobs_by_source.items():
            if source not in self.collectors:
                logger.warning(f"No collector for source: {source}")
                continue
            
            collector = self.collectors[source]
            
            # Process in batches to avoid overwhelming the sites
            for i, job in enumerate(source_jobs):
                if not job.url or not job.id:
                    continue
                
                # Skip if description is already substantial (> 500 chars)
                if job.description and len(job.description) > 500:
                    logger.debug(f"Job {job.id} already has description ({len(job.description)} chars)")
                    continue
                
                try:
                    description = await collector.fetch_job_description(job.url)
                    if description:
                        results[job.id] = description
                        logger.debug(
                            f"Fetched description for job {job.id}: "
                            f"{len(description)} chars"
                        )
                    
                    # Progress logging
                    if (i + 1) % 10 == 0:
                        logger.info(
                            f"Description fetch progress ({source}): "
                            f"{i + 1}/{len(source_jobs)}"
                        )
                    
                    # Small delay between fetches
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Error fetching description for job {job.id}: {e}")
                    continue
        
        logger.info(f"Fetched {len(results)} full descriptions")
        return results
    
    async def close(self) -> None:
        """Clean up all resources."""
        for collector in self.collectors.values():
            await collector.close()
        
        if self.browser:
            await self.browser.close()
        
        if self.playwright:
            await self.playwright.stop()
        
        logger.info("Collector manager closed")


def raw_to_job(raw: RawJobListing, existing_hash: Optional[str] = None) -> Job:
    """
    Convert a RawJobListing to a Job object.
    
    Args:
        raw: The raw job listing.
        existing_hash: Optional pre-computed hash.
    
    Returns:
        Job object ready for database storage.
    """
    job_hash = existing_hash or Job.create_hash(raw.title, raw.company, raw.url)
    
    return Job(
        id=None,
        job_hash=job_hash,
        title=raw.title,
        company=raw.company,
        location=raw.location,
        description=raw.description,
        url=raw.url,
        source=raw.source,
        latitude=None,
        longitude=None,
        is_remote=False,
        distance_miles=None,
        matched_city=None,
        semantic_score=None,
        score_reasoning=None,
        application_status=ApplicationStatus.NEW,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        raw_data=str(raw.raw_data) if raw.raw_data else None,
    )


# Import ApplicationStatus for the raw_to_job function
from .database import ApplicationStatus
