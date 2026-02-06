#!/usr/bin/env python3
"""
Test script for pay range extraction.

Tests the pay_parser module against various real-world pay range formats.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semanticscout.pay_parser import extract_pay_range, PayType

# Test cases from the provided pay.txt file
test_cases = [
    # Standard salary ranges with em-dash
    ("$149,000—$157,000 USD", 149000, 157000, PayType.SALARY),
    ("$142,000—$150,000 USD", 142000, 150000, PayType.SALARY),
    ("$103,100—$172,000 USD", 103100, 172000, PayType.SALARY),
    ("$202,000—$278,000 USD", 202000, 278000, PayType.SALARY),
    
    # Standard salary ranges with hyphen
    ("$100,000.00 - $140,000.00", 100000, 140000, PayType.SALARY),
    ("$138,000—$190,900 USD", 138000, 190900, PayType.SALARY),
    
    # K notation
    ("The base salary range for this role is $125.6K to $157K.", 125600, 157000, PayType.SALARY),
    ("Our client is seeking an Oracle Benefits HRIS Analyst. This is a full-time, direct hire role; 100% remote (working PST hours). Salary Range for the Role is $100k to $145k.", 100000, 145000, PayType.SALARY),
    
    # Ranges with full text context
    ("The expected base salary range for this position is $115,000.00-$135,000.00", 115000, 135000, PayType.SALARY),
    ("The US base salary range for this full-time position is $130,000 - $194,000 .", 130000, 194000, PayType.SALARY),
    ("Compensation Range: $120,000 - $205,200", 120000, 205200, PayType.SALARY),
    ("$146,561—$301,744 USD", 146561, 301744, PayType.SALARY),
    
    # Hourly rates
    ("The pay range for this position is $27-$31 per hour, with starting pay based on location.", 27, 31, PayType.HOURLY),
    ("Pay Range: $16-$25 per hr. depending on experience", 16, 25, PayType.HOURLY),
    ("Hourly pay rate: : $104.90-$111.90", 104.90, 111.90, PayType.HOURLY),
    ("$60/Hr. - $65/Hr. (on W2, all-inclusive)", 60, 65, PayType.HOURLY),
    ("Rate: $70 -$75 per hour", 70, 75, PayType.HOURLY),
    ("This is a remote contractor position, with an estimated 25-35 hours per week. $26-$32 per hour.", 26, 32, PayType.HOURLY),
    
    # More complex text with salary
    ("The estimated base salary range for this job is $120,000 - $145,000.", 120000, 145000, PayType.SALARY),
    ("Salary Range: $87,600 - $117,100", 87600, 117100, PayType.SALARY),
    ("US Base Salary Range: $86,600.00 - $112,500.00", 86600, 112500, PayType.SALARY),
    
    # Ranges with "to" instead of dash
    ("The compensation range for this position ranges from $72,750.00 - $151,250.00/yr", 72750, 151250, PayType.SALARY),
    
    # Very wide ranges
    ("The salary range for this position is: $91,000 - $321,500.", 91000, 321500, PayType.SALARY),
    
    # Location-specific ranges (should still extract the main range)
    ("In California, Connecticut, Maryland, Massachusetts, New Jersey, New York, Washington state, and Washington DC the standard base pay range for this role is $137,100.00 - $218,900.00 annually.", 137100, 218900, PayType.SALARY),
]

def test_pay_extraction():
    """Run all test cases."""
    passed = 0
    failed = 0
    
    for text, expected_min, expected_max, expected_type in test_cases:
        result = extract_pay_range(text)
        
        if result is None:
            print(f"❌ FAILED (no match): {text[:60]}...")
            failed += 1
            continue
        
        # Check values with some tolerance for floating point
        min_ok = abs(result.min_amount - expected_min) < 1
        max_ok = abs(result.max_amount - expected_max) < 1
        type_ok = result.pay_type == expected_type
        
        if min_ok and max_ok and type_ok:
            print(f"✅ PASSED: {result.format_display()} <- {text[:50]}...")
            passed += 1
        else:
            print(f"❌ FAILED: Got {result.format_display()} (type={result.pay_type.value}), " +
                  f"expected ${expected_min:,.0f}-${expected_max:,.0f} ({expected_type.value})")
            print(f"   Text: {text[:80]}...")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    
    return failed == 0


if __name__ == "__main__":
    success = test_pay_extraction()
    sys.exit(0 if success else 1)
