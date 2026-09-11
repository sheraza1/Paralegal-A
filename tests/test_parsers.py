import math
import re
import sys
import os
import types

import pytest

# Import parsing helpers from generate_report.py
from generate_report import (
    _parse_paid_to_date_from_medical_sections,
    _parse_future_med_from_medical_sections,
    _parse_total_wage_loss_from_sections,
    _parse_total_time_off_weeks_from_wage_summary,
    _parse_future_wage_breakdown_from_sections,
    _parse_property_damage_breakdown_from_sections,
    _sanitize_police_for_facts,
    _parse_average_annual_earning_from_sections,
)


def test_parse_paid_to_date_from_medical_sections():
    text = (
        "TOTAL MEDICAL EXPENSES SUMMARY\n"
        "Paid to Date: $8,950\n"
        "Emergency Department: $1,200\n"
        "Future Medical Expenses (estimated): $2,000-4,000 over next 5 years.\n"
    )
    assert _parse_paid_to_date_from_medical_sections(text) == 8950.0


def test_parse_future_med_from_medical_sections_range_uses_upper_ceiled():
    text = (
        "TOTAL MEDICAL EXPENSES SUMMARY\n"
        "Future Medical Expenses (estimated): $2,000-4,000 over next 5 years.\n"
    )
    assert _parse_future_med_from_medical_sections(text) == 4000.0


def test_parse_total_wage_loss_from_sections():
    text = (
        "WAGE LOSS SUMMARY\n"
        "Hourly Rate: $22.00\n"
        "Total Time Off Work: 10.5 weeks\n"
        "Total Wage Loss: $4,940\n"
    )
    assert _parse_total_wage_loss_from_sections(text) == 4940.0


def test_parse_total_time_off_weeks_from_wage_summary():
    text = (
        "WAGE LOSS SUMMARY\n"
        "Total Time Off Work: 10.5 weeks\n"
    )
    assert _parse_total_time_off_weeks_from_wage_summary(text) == 10.5


def test_parse_future_wage_breakdown_from_sections_prefers_upper_annual():
    text = (
        "FUTURE WAGE IMPACT ANALYSIS\n"
        "Annual Loss: $5,000 - $6,500\n"
    )
    total, bullets = _parse_future_wage_breakdown_from_sections(text)
    assert total == 6500.0


def test_parse_property_damage_breakdown_from_sections_vehicle1_victim_upper_ceiled():
    text = (
        "PROPERTY DAMAGE\n"
        "Vehicle #1 (Victim):\n"
        "Damage: Moderate to severe rear-end damage\n"
        "Estimated Repair Cost: $8,500 - $12,000\n"
    )
    amt, bullets = _parse_property_damage_breakdown_from_sections(text, plaintiff_name="John Smith")
    assert amt == 12000.0
    assert any("Repair" in b or "Estimate" in b for b in bullets)


def test_sanitize_police_for_facts_removes_pd_block_and_codes():
    raw = (
        "NARRATIVE\n"
        "Driver stated he looked down.\n"
        "PROPERTY DAMAGE\n"
        "Vehicle #1 (Victim) damage details...\n"
        "Estimated Repair Cost: $8,500 - $12,000\n"
        "\n"
        "Officer issued citation under CVC 21453.\n"
    )
    cleaned = _sanitize_police_for_facts(raw)
    assert "PROPERTY DAMAGE" not in cleaned
    assert "Estimated Repair" not in cleaned
    assert "CVC" not in cleaned


def test_parse_average_annual_earning_from_sections():
    """Test parsing Average Annual Earning from EMPLOYMENT VERIFICATION LETTER section."""
    # Test case with parentheses
    text1 = "Average Annual Earnings (based on 2023): $84,500"
    assert _parse_average_annual_earning_from_sections(text1) == 84500.0
    
    # Test case without parentheses
    text2 = "Average Annual Earnings: $84,500"
    assert _parse_average_annual_earning_from_sections(text2) == 84500.0
    
    # Test case with different formatting
    text3 = "Average Annual Earnings: $84,500.00"
    assert _parse_average_annual_earning_from_sections(text3) == 84500.0
    
    # Test case with no match
    text4 = "No earnings information here"
    assert _parse_average_annual_earning_from_sections(text4) is None
    
    # Test case with empty text
    assert _parse_average_annual_earning_from_sections("") is None
    assert _parse_average_annual_earning_from_sections(None) is None 