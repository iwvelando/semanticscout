"""
Pay range extraction module for Semantic Scout.

Extracts salary/hourly compensation information from job descriptions
with support for various formats including K notation, ranges, and hourly rates.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class PayType(Enum):
    """Type of compensation."""
    SALARY = "salary"
    HOURLY = "hourly"
    UNKNOWN = "unknown"


@dataclass
class PayRange:
    """Represents an extracted pay range."""
    min_amount: float
    max_amount: float
    pay_type: PayType
    raw_text: str  # Original matched text for reference
    
    def format_amount(self, amount: float) -> str:
        """Format a single amount for display."""
        if self.pay_type == PayType.HOURLY:
            # Format hourly with cents if present
            if amount == int(amount):
                return f"${int(amount)}"
            return f"${amount:.2f}"
        else:
            # Format salary in K notation
            if amount >= 1000:
                k_amount = amount / 1000
                if k_amount == int(k_amount):
                    return f"${int(k_amount)}K"
                return f"${k_amount:.1f}K"
            return f"${amount:,.0f}"
    
    def format_display(self) -> str:
        """Format the pay range for display."""
        min_str = self.format_amount(self.min_amount)
        max_str = self.format_amount(self.max_amount)
        
        type_suffix = "/hr" if self.pay_type == PayType.HOURLY else "/yr"
        
        if self.min_amount == self.max_amount:
            return f"{min_str}{type_suffix}"
        return f"{min_str} - {max_str}{type_suffix}"
    
    def format_slack(self) -> str:
        """Format the pay range for Slack display."""
        return f"💰 {self.format_display()}"
    
    def __str__(self) -> str:
        return self.format_display()


def extract_pay_range(description: str) -> Optional[PayRange]:
    """
    Extract pay range from a job description.
    
    Handles various formats:
    - $100,000 - $150,000
    - $100K - $150K (case insensitive)
    - $100,000.00 - $150,000.00
    - $25 - $35 per hour
    - $100,000—$150,000 (em dash)
    - $103,100—$172,000 USD
    
    Args:
        description: The job description text.
        
    Returns:
        PayRange if found, None otherwise.
    """
    if not description:
        return None
    
    # Normalize different dash types to standard hyphen
    normalized = description.replace('—', '-').replace('–', '-')
    
    # Try hourly patterns first (more specific)
    hourly_range = _extract_hourly_range(normalized, description)
    if hourly_range:
        return hourly_range
    
    # Then try salary patterns
    salary_range = _extract_salary_range(normalized, description)
    if salary_range:
        return salary_range
    
    return None


def _extract_hourly_range(normalized: str, original: str) -> Optional[PayRange]:
    """
    Extract hourly pay range from text.
    
    Patterns:
    - $25 - $35 per hour
    - $25/hr - $35/hr
    - $25.50 - $35.00 hourly
    - Hourly pay rate: $104.90-$111.90
    """
    # Pattern for hourly rates with explicit "hour/hr/hourly" indicator
    # Look for ranges like "$25 - $35 per hour" or "$25/hr"
    patterns = [
        # Range with "per hour", "hourly", "/hr" after
        r'\$(\d{1,3}(?:\.\d{1,2})?)\s*[-–—]\s*\$(\d{1,3}(?:\.\d{1,2})?)\s*(?:per\s+h(?:ou)?r|hourly|/\s*h(?:ou)?r)',
        # Range with hourly indicator before
        r'(?:hourly\s+(?:pay\s+)?(?:rate|range)?[:\s]*)\s*\$(\d{1,3}(?:\.\d{1,2})?)\s*[-–—]\s*\$(\d{1,3}(?:\.\d{1,2})?)',
        # Range with /Hr. or /hr on each value (e.g., $60/Hr. - $65/Hr.)
        r'\$(\d{1,3}(?:\.\d{1,2})?)\s*/\s*[Hh](?:ou)?r\.?\s*[-–—]\s*\$(\d{1,3}(?:\.\d{1,2})?)\s*/\s*[Hh](?:ou)?r\.?',
        # Pay range with "per hr" or similar
        r'(?:pay\s+range[:\s]*)\s*\$(\d{1,3}(?:\.\d{1,2})?)\s*[-–—]\s*\$(\d{1,3}(?:\.\d{1,2})?)\s*per\s+h',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, normalized, re.IGNORECASE)
        if match:
            min_val = _parse_number(match.group(1))
            max_val = _parse_number(match.group(2))
            
            # Sanity check for hourly (usually $10-$500/hr range)
            if min_val and max_val and 5 <= min_val <= 500 and 5 <= max_val <= 500:
                # Find the original text for context
                raw_text = _find_original_context(original, match.group(0))
                return PayRange(
                    min_amount=min(min_val, max_val),
                    max_amount=max(min_val, max_val),
                    pay_type=PayType.HOURLY,
                    raw_text=raw_text
                )
    
    return None


def _extract_salary_range(normalized: str, original: str) -> Optional[PayRange]:
    """
    Extract annual salary range from text.
    
    Patterns:
    - $100,000 - $150,000
    - $100K - $150K
    - $100,000.00 - $150,000.00
    - $103,100—$172,000 USD
    - Base salary range: $125.6K to $157K
    """
    patterns = [
        # K notation: $125.6K - $157K (case insensitive)
        r'\$(\d{1,3}(?:\.\d{1,2})?)\s*[kK]\s*(?:[-–—]|to)\s*\$(\d{1,3}(?:\.\d{1,2})?)\s*[kK]',
        # Full numbers with commas: $100,000 - $150,000
        r'\$(\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?)\s*(?:[-–—]|to)\s*\$(\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?)',
        # Numbers without commas but clearly salaries (5+ digits): $100000 - $150000
        r'\$(\d{5,}(?:\.\d{1,2})?)\s*(?:[-–—]|to)\s*\$(\d{5,}(?:\.\d{1,2})?)',
        # Mixed K and full: $100K - $150,000 (less common but handle it)
        r'\$(\d{1,3}(?:\.\d{1,2})?)\s*[kK]\s*(?:[-–—]|to)\s*\$(\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?)',
        r'\$(\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?)\s*(?:[-–—]|to)\s*\$(\d{1,3}(?:\.\d{1,2})?)\s*[kK]',
    ]
    
    best_match = None
    best_range = None
    
    for pattern in patterns:
        for match in re.finditer(pattern, normalized, re.IGNORECASE):
            val1_str = match.group(1)
            val2_str = match.group(2)
            
            # Parse the values
            min_val = _parse_salary_value(val1_str, pattern)
            max_val = _parse_salary_value(val2_str, pattern)
            
            if min_val is None or max_val is None:
                continue
            
            # Sanity check for annual salary (usually $20k-$1M range)
            if not (15000 <= min_val <= 2000000 and 15000 <= max_val <= 2000000):
                continue
            
            # Skip if this looks like it might be hourly (check surrounding context)
            context_start = max(0, match.start() - 50)
            context_end = min(len(normalized), match.end() + 50)
            context = normalized[context_start:context_end].lower()
            
            if any(h in context for h in ['per hour', 'hourly', '/hr', '/hour', 'per hr']):
                continue
            
            # Prefer ranges that are more complete (have more context words)
            score = 0
            if 'salary' in context or 'base pay' in context or 'compensation' in context:
                score += 10
            if 'annual' in context or 'yearly' in context or 'year' in context:
                score += 5
            if 'usd' in context:
                score += 2
            
            if best_match is None or score > best_match:
                best_match = score
                raw_text = _find_original_context(original, match.group(0))
                best_range = PayRange(
                    min_amount=min(min_val, max_val),
                    max_amount=max(min_val, max_val),
                    pay_type=PayType.SALARY,
                    raw_text=raw_text
                )
    
    return best_range


def _parse_number(value_str: str) -> Optional[float]:
    """Parse a simple number string."""
    try:
        # Remove commas and parse
        cleaned = value_str.replace(',', '')
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def _parse_salary_value(value_str: str, pattern: str) -> Optional[float]:
    """
    Parse a salary value string, handling K notation.
    
    Args:
        value_str: The value string (e.g., "125.6" for K notation, "125,600" for full)
        pattern: The regex pattern used (to determine if K notation)
    """
    try:
        # Remove commas
        cleaned = value_str.replace(',', '')
        value = float(cleaned)
        
        # Check if this pattern is K notation (has [kK] in it near the capture group)
        if '[kK]' in pattern or 'K' in pattern.upper():
            # If value is small (< 1000), it's probably K notation
            if value < 1000:
                return value * 1000
        
        return value
    except (ValueError, TypeError):
        return None


def _find_original_context(original: str, match_text: str) -> str:
    """
    Find the original text context around a match.
    
    Tries to capture the full sentence or line containing the match.
    """
    # Normalize the match text for searching
    normalized_match = match_text.replace('—', '-').replace('–', '-')
    normalized_original = original.replace('—', '-').replace('–', '-')
    
    idx = normalized_original.find(normalized_match)
    if idx == -1:
        return match_text
    
    # Find sentence boundaries (. or newline)
    start = idx
    end = idx + len(match_text)
    
    # Expand to sentence start
    while start > 0 and original[start - 1] not in '.\n':
        start -= 1
        if idx - start > 150:  # Limit context
            break
    
    # Expand to sentence end
    while end < len(original) and original[end] not in '.\n':
        end += 1
        if end - idx > 150:  # Limit context
            break
    
    return original[start:end].strip()


def format_pay_for_report(pay_range: Optional[PayRange]) -> str:
    """
    Format pay range for markdown report.
    
    Args:
        pay_range: The extracted pay range, or None.
        
    Returns:
        Formatted string for report, or "Not specified" if None.
    """
    if pay_range is None:
        return "Not specified"
    return pay_range.format_display()


def format_pay_for_slack(pay_range: Optional[PayRange]) -> Optional[str]:
    """
    Format pay range for Slack display.
    
    Args:
        pay_range: The extracted pay range, or None.
        
    Returns:
        Formatted string for Slack, or None if no pay range.
    """
    if pay_range is None:
        return None
    return pay_range.format_slack()
