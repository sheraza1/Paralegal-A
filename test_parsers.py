from generate_report import (
    _parse_currency_amounts,
    _sum_from_citations,
    _extract_weeks_from_citations,
    _parse_total_wage_loss_from_sections,
    _parse_average_annual_earning_from_sections,
)

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